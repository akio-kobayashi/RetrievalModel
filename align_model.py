import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.checkpoint import checkpoint
#from kornia.losses import SoftDTW

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

    all_inf = (mask == float("-inf")).all(dim=-1, keepdim=True)
    mask = mask.masked_fill(all_inf, 0.0)
    
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

        '''
        # 入力に NaN がないかチェック
        print(f"[DEBUG][decode] src_hubert NaN? {src_hubert.isnan().any().item()}, "
              f"src_pitch NaN? {src_pitch.isnan().any().item()}")
        '''
        # --- Encode ---
        x = self.hubert_proj(src_hubert) + self.pitch_proj(src_pitch.unsqueeze(-1))
        #print(f"[DEBUG][encode] after proj x NaN? {x.isnan().any().item()}")

        for i, layer in enumerate(self.encoder_layers):
            # self-attn + FFN
            x2, _ = layer['self_attn'](x, x, x)
            x = x + x2
            x = layer['ffn'](x)
            #print(f"[DEBUG][encode][layer {i} self_attn] x NaN? {x.isnan().any().item()}")

            # pitch-attn + FFN
            p_stream = self.pitch_proj(src_pitch.unsqueeze(-1))
            x2p, _ = layer['pitch_attn'](x, p_stream, p_stream)
            x = x + x2p
            x = layer['ffn'](x)
            #print(f"[DEBUG][encode][layer {i} pitch_attn] x NaN? {x.isnan().any().item()}")

        memory = x
        #print(f"[DEBUG][encode] final memory NaN? {memory.isnan().any().item()}")

        # --- Decode ---
        current = self.bos_token.expand(B, 1, self.d_model)
        #print(f"[DEBUG][decode] start current NaN? {current.isnan().any().item()}")

        decoded_h, decoded_p = [], []
        ended = torch.zeros(B, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            t_len = current.size(1)
            mask = compute_diagonal_mask(t_len, S, self.nu, device)
            mask = safe_attn_mask(mask)
            x = current
            #print(f"[DEBUG][decode][step {t}] before layers x NaN? {x.isnan().any().item()}")

            for j, layer in enumerate(self.decoder_layers):
                # self-attn + FFN
                x2, _ = layer['self_attn'](x, x, x)
                x = x + x2
                x = layer['ffn'](x)
                #print(f"[DEBUG][decode][step {t}][layer {j} self_attn] NaN? {x.isnan().any().item()}")
                
                # pitch-attn + FFN
                if decoded_p:
                    pitch_list = torch.cat(decoded_p, dim=1).unsqueeze(-1)
                else:
                    pitch_list = torch.zeros(B, 0, 1, device=device)
                pitch_bos = torch.zeros(B, 1, self.d_model, device=device)
                pitch_stream = torch.cat([pitch_bos, self.pitch_proj(pitch_list)], dim=1)

                x2p, _ = layer['pitch_attn'](x, pitch_stream, pitch_stream)
                x = x + x2p
                x = layer['ffn'](x)
                #print(f"[DEBUG][decode][step {t}][layer {j} pitch_attn] NaN? {x.isnan().any().item()}")

                # cross-attn + FFN
                x2m, _ = layer['cross_attn'](x, memory, memory, attn_mask=mask)
                x = x + x2m
                x = layer['ffn'](x)
                #if not torch.isfinite(x).all():
                #    print(f"[DEBUG][decode][step {t}][layer {j} cross_attn] contains NaN or inf")

            # 最後の時刻を取り出して出力
            last = x[:, -1, :]
            logits = self.token_classifier(last)
            pred_cls = logits.argmax(-1)
            h_pred = self.out_hubert(last)
            p_pred = self.out_pitch(last).squeeze(-1)

            '''
            if not torch.isfinite(h_pred).all():
                print(f"[DEBUG][decode][step {t}] h_pred contains NaN or inf, mean={h_pred.mean().item()}")
            if not torch.isfinite(p_pred).all():
                print(f"[DEBUG][decode][step {t}] p_pred contains NaN or inf, mean={p_pred.mean().item()}")
            '''
            decoded_h.append(h_pred.unsqueeze(1))
            decoded_p.append(p_pred.unsqueeze(1))

            # 次の入力用に融合
            fused = self.hubert_proj(h_pred) + self.pitch_proj(p_pred.unsqueeze(-1))
            '''
            if not torch.isfinite(fused).all():
                print(f"[DEBUG][decode][step {t}] fused contains NaN or inf, "
                      f"hub_proj mean={(self.hubert_proj(h_pred)).mean().item()}, "
                      f"pit_proj mean={(self.pitch_proj(p_pred.unsqueeze(-1))).mean().item()}")
            '''
            current = torch.cat([current, fused.unsqueeze(1)], dim=1)
            ended = ended | (pred_cls == 1)
            if ended.all():
                break

            hubert_seq = torch.cat(decoded_h, dim=1)
            pitch_seq = torch.cat(decoded_p, dim=1)
            return hubert_seq, pitch_seq
