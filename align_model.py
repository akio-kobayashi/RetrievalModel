import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint
#from kornia.losses import SoftDTW

class FixedPositionalEncoding(nn.Module):
    """学習しない正弦位置エンコーダ（重み互換を保つ）"""
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)   # ← buffer なので state_dict に入らない

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[start : start + x.size(1)]

# ----------------------------------------------------------------
# Diagonal mask function (VoiceTransformer style)
# ----------------------------------------------------------------
def compute_diagonal_mask(T: int, S: int, nu: float = 0.3, device=None) -> torch.Tensor:
    """
    Returns an additive attention mask of shape (T, S) that biases attention
    towards the diagonal using a Gaussian kernel.
    """
    tgt = torch.arange(T, device=device).unsqueeze(1).float() / (T - 1)
    src = torch.arange(S, device=device).unsqueeze(0).float() / (S - 1)
    diff = (src - tgt) ** 2
    weight = torch.exp(-diff / (2 * nu * nu))
    mask = torch.log(weight)
    mask = torch.clamp(mask, min=-1e9)

    # ---- 追加：行がすべて -1e9 なら中心列だけ 0.0 にする ----
    bad_row = (mask == -1e9).all(dim=-1)         # shape (T,)
    if bad_row.any():
        rows = bad_row.nonzero(as_tuple=False).squeeze(1)       # 行インデックス
        cols = ((src - tgt).abs().argmin(dim=-1))[rows]         # その行の最近接列
        mask[rows, cols] = 0.0                                  # 行ごとに 1 列だけ 0.0        
        #center_idx = (torch.arange(S, device=device) / (S - 1)).unsqueeze(0)  # [1,S]
        #center_idx = (center_idx - tgt).abs().argmin(dim=-1)                  # 最近接列
        #mask[bad_row, :] = -1e9                                               # まず全部 -1e9
        #mask[bad_row, center_idx[bad_row]] = 0.0                              # 中央 1 列だけ 0
    # -----------------------------------------------------------

    #all_inf = (mask == float("-inf")).all(dim=-1, keepdim=True)
    #mask = mask.masked_fill(all_inf, 0.0)
    
    return mask

def safe_attn_mask(mask: torch.Tensor, neg_inf: float = -1e4) -> torch.Tensor:
    if mask is None:
        return None

    if mask.dtype == torch.bool:
        return mask  # True/False マスクならそのまま使う

    elif mask.dtype in (torch.float32, torch.float64):
        mask = mask.clone()

        # NaN → neg_inf
        mask[mask != mask] = neg_inf

        # -inf → neg_inf
        mask[mask == float('-inf')] = neg_inf

        # all-inf rows → 0.0
        all_inf = (mask == neg_inf).all(dim=-1, keepdim=True)
        mask = mask.masked_fill(all_inf, 0.0)

        return mask

    else:
        raise ValueError(f"Unsupported attn_mask dtype: {mask.dtype}")
    
# ----------------------------------------------------------------
# Transformer-based Aligner Module with Modality Cross-Attention and Diagonal Loss
# ----------------------------------------------------------------
class TransformerAligner(nn.Module):
    def __init__(
        self,
        input_dim_hubert: int = 768,
        input_dim_pitch: int = 1,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nu: float = 0.3,
        diag_weight: float = 1.0,
        ce_weight: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.nu = nu
        self.diag_weight = diag_weight
        self.ce_weight = ce_weight

        # Shared projections
        self.hubert_proj = nn.Linear(input_dim_hubert, d_model)
        self.pitch_proj  = nn.Linear(input_dim_pitch, d_model)

        self.posenc = FixedPositionalEncoding(self.d_model, max_len=8000)
        
        # Encoder layers with modality fusion
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'pitch_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                )
            }))

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'pitch_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                )
            }))

        # Output heads
        self.out_hubert = nn.Linear(d_model, input_dim_hubert)
        self.out_pitch  = nn.Linear(d_model, input_dim_pitch)
        self.token_classifier = nn.Linear(d_model, 2)

        # EOS/BOS tokens
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.eos_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, src_hubert, src_pitch, tgt_hubert, tgt_pitch):
        B, S, _ = src_hubert.size()
        _, T, _ = tgt_hubert.size()
        device = src_hubert.device

        # Input fusion
        x = self.hubert_proj(src_hubert) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)
        
        # Precompute pitch stream for encoder
        p_stream_enc = self.pitch_proj(src_pitch.unsqueeze(-1))  # (B, S, d_model)

        # Encoder pass with checkpoint
        def encoder_block(x, p_stream, layer):
            x2, _ = layer['self_attn'](x, x, x, need_weights=False)
            x_ = layer['ffn'](x + x2)
            x2p, _ = layer['pitch_attn'](x_, p_stream, p_stream, need_weights=False)
            return layer['ffn'](x_ + x2p)

        for layer in self.encoder_layers:
            x = checkpoint(encoder_block, x, p_stream_enc, layer)

        memory = x

        # Decoder initialization
        bos = self.bos_token.expand(B, 1, self.d_model)
        tgt_h = self.hubert_proj(tgt_hubert)
        tgt_p = self.pitch_proj(tgt_pitch.unsqueeze(-1))
        tgt_fuse = tgt_h + tgt_p
        x = torch.cat([bos, tgt_fuse], dim=1)
        x = self.posenc(x)
        
        # Prepare pitch stream decoder
        pitch_only = tgt_p
        pitch_bos = torch.zeros(B,1,self.d_model, device=device)
        p_stream_dec = torch.cat([pitch_bos, pitch_only], dim=1)  # (B, T+1, d_model)

        mask = compute_diagonal_mask(T+1, S, self.nu, device)
        # Decoder pass with checkpoint
        def decoder_block(x, p_stream, memory, layer, mask):
            y2, _ = layer['self_attn'](x, x, x, need_weights=False)
            y_ = layer['ffn'](x + y2)
            y2p, _ = layer['pitch_attn'](y_, p_stream, p_stream, need_weights=False)
            y__ = layer['ffn'](y_ + y2p)
            y2m, attn_w = layer['cross_attn'](y__, memory, memory, attn_mask=mask, need_weights=True)
            return layer['ffn'](y__ + y2m), attn_w

        attn_w = None
        for layer in self.decoder_layers:
            x, attn_w = checkpoint(decoder_block, x, p_stream_dec, memory, layer, mask)

        # Predictions
        pred_hubert = self.out_hubert(x)
        pred_pitch  = self.out_pitch(x).squeeze(-1)
        logits      = self.token_classifier(x)

        # for NaN 
        if torch.isnan(attn_w).any():
            attn_w = torch.nan_to_num(attn_w)
        if torch.isnan(pred_hubert).any():
            pred_hubert = torch.nan_to_num(pred_hubert)
        if torch.isnan(pred_pitch).any():
            pred_pitch = torch.nan_to_num(pred_pitch)

        # Loss
        tgt_h_pad = torch.cat([tgt_hubert, torch.zeros(B,1,tgt_hubert.size(-1),device=device)], dim=1)
        tgt_p_pad = torch.cat([tgt_pitch.unsqueeze(-1), torch.zeros(B,1,1,device=device)], dim=1).squeeze(-1)
        loss_h = F.l1_loss(pred_hubert, tgt_h_pad)
        loss_p = F.l1_loss(pred_pitch, tgt_p_pad)
        # Diagonal loss
        pos_s = torch.arange(S, device=device).unsqueeze(0).repeat(T+1,1)
        pos_t = torch.arange(T+1, device=device).unsqueeze(1).repeat(1,S)
        dist = torch.abs(pos_t - pos_s).float() / S
        loss_diag = (attn_w * dist.unsqueeze(0)).sum() / (B * (T+1)) 
        # EOS CE
        labels = torch.zeros(B, T+1, dtype=torch.long, device=device); labels[:,-1]=1
        logits = torch.clamp(logits, -20.0, 20.0)
        loss_ce = F.cross_entropy(logits.view(-1,2), labels.view(-1), label_smoothing=0.1)

        total = loss_h + loss_p \
              + self.diag_weight * loss_diag \
              + self.ce_weight*loss_ce
        if torch.isnan(total):
            total = torch.nan_to_num(total)

        self.last_preds = {
            "hubert_pred": pred_hubert,
            "pitch_pred":  pred_pitch
        }
        
        return total, {
            'hubert_l1': loss_h,
            'pitch_l1': loss_p,
            'diag': loss_diag,
            'ce': loss_ce
        }

    def greedy_decode(self, src_hubert, src_pitch, max_len=200):
        B, S, _ = src_hubert.size()
        device = src_hubert.device
        
        # --- Encode ---
        x = self.hubert_proj(src_hubert) + self.pitch_proj(src_pitch.unsqueeze(-1))
        x = self.posenc(x)        
        for layer in self.encoder_layers:
            x = layer['ffn'](x + layer['self_attn'](x, x, x)[0])

            p_stream = self.pitch_proj(src_pitch.unsqueeze(-1))
            x = layer['ffn'](x + layer['pitch_attn'](x, p_stream, p_stream)[0])

            #memory = x  # (B, S, d_model)
        memory = x
        
        # --- Decode loop ---
        current  = self.bos_token.expand(B, 1, self.d_model)
        decoded_h, decoded_p = [], []
        ended = torch.zeros(B, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            # mask などはそのまま
            x = current
            for layer in self.decoder_layers:
                x = layer['ffn'](x + layer['self_attn'](x, x, x)[0])
                
                pitch_list = torch.cat(decoded_p, dim=1).unsqueeze(-1) if decoded_p else torch.zeros(B, 0, 1, device=device)
                pitch_stream = torch.cat([torch.zeros(B, 1, self.d_model, device=device),
                                          self.pitch_proj(pitch_list)], dim=1)
                
                x = layer['ffn'](x + layer['pitch_attn'](x, pitch_stream, pitch_stream)[0])
                float_mask = compute_diagonal_mask(
                    current.size(1),
                    memory.size(1),
                    self.nu,
                    device
                )
                # -1e9 と 0.0 だけを保持
                float_mask = safe_attn_mask(float_mask, neg_inf=-1e9)

                # bool マスク（True=mask, False=keep）へ変換
                bool_mask = (float_mask == -1e9)                
                x2m, _ = layer['cross_attn'](
                    x,
                    memory,
                    memory,
                    attn_mask=bool_mask
                )
                x = layer['ffn'](x + x2m)
                
            last   = x[:, -1, :]
            h_pred = self.out_hubert(last)           # (B, hubert_dim)
            p_pred = self.out_pitch(last).squeeze(-1)  # (B,)

            decoded_h.append(h_pred.unsqueeze(1))
            decoded_p.append(p_pred.unsqueeze(1))

            fused = self.hubert_proj(h_pred) + self.pitch_proj(p_pred.unsqueeze(-1))
            fused = self.posenc(fused.unsqueeze(1), start=current.size(1)).squeeze(1)
            current = torch.cat([current, fused.unsqueeze(1)], dim=1)

            ended |= (self.token_classifier(last).argmax(-1) == 1)
            if ended.all():
                break

        # ----- ここでまとめて return -----
        hubert_seq = torch.cat(decoded_h, dim=1)   # (B, T, hubert_dim)
        pitch_seq  = torch.cat(decoded_p, dim=1)   # (B, T)
        return hubert_seq, pitch_seq
    
