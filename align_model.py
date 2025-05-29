import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from soft_dtw import SoftDTW

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
    weight = torch.exp(-diff / (2 * nu * nu))  # Gaussian kernel
    mask = torch.log(weight)  # additive log-probabilities
    return mask

# ----------------------------------------------------------------
# Transformer-based Aligner Module with Diagonal Attention Mask
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
        sdtw_weight: float = 1.0,
        ce_weight: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.nu = nu
        self.sdtw_weight = sdtw_weight
        self.ce_weight = ce_weight

        # Input projections
        self.src_hubert_proj = nn.Linear(input_dim_hubert, d_model)
        self.src_pitch_proj  = nn.Linear(input_dim_pitch, d_model)
        self.tgt_hubert_proj = nn.Linear(input_dim_hubert, d_model)
        self.tgt_pitch_proj  = nn.Linear(input_dim_pitch, d_model)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])

        # Output heads
        self.out_hubert = nn.Linear(d_model, input_dim_hubert)
        self.out_pitch  = nn.Linear(d_model, input_dim_pitch)
        self.token_classifier = nn.Linear(d_model, 2)

        # EOS/BOS tokens
        self.bos_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.eos_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Soft-DTW
        self.sdtw = SoftDTW(use_cuda=torch.cuda.is_available())

    def forward(self, src_hubert, src_pitch, tgt_hubert, tgt_pitch):
        B, S, _ = src_hubert.size()
        _, T, _ = tgt_hubert.size()
        device = src_hubert.device

        # Fuse inputs
        src = self.src_hubert_proj(src_hubert) + self.src_pitch_proj(src_pitch.unsqueeze(-1))
        tgt = self.tgt_hubert_proj(tgt_hubert) + self.tgt_pitch_proj(tgt_pitch.unsqueeze(-1))

        # Encode
        memory = self.encoder(src)

        # Prepare decoder input
        bos = self.bos_token.expand(B, 1, self.d_model)
        tgt_in = torch.cat([bos, tgt], dim=1)  # (B, T+1, d_model)

        # Precompute diagonal mask
        mask = compute_diagonal_mask(T+1, S, self.nu, device)  # (T+1, S)

        # Decode
        x = tgt_in
        attn_w = None
        for layer in self.decoder_layers:
            x2, _ = layer['self_attn'](x, x, x)
            x = x + x2; x = layer['ffn'](x)
            x2, attn_w = layer['cross_attn'](
                x, memory, memory,
                attn_mask=mask  # biases cross-attention
            )
            x = x + x2; x = layer['ffn'](x)

        # Outputs
        pred_hubert = self.out_hubert(x)            # (B, T+1, H_dim)
        pred_pitch  = self.out_pitch(x).squeeze(-1)  # (B, T+1)
        logits      = self.token_classifier(x)      # EOS token logits

        # Pad targets with EOS placeholder
        tgt_h_pad = torch.cat([tgt_hubert, torch.zeros(B,1,tgt_hubert.size(-1),device=device)], dim=1)
        tgt_p_pad = torch.cat([tgt_pitch.unsqueeze(-1), torch.zeros(B,1,1,device=device)], dim=1).squeeze(-1)

        # Losses
        loss_h = F.l1_loss(pred_hubert, tgt_h_pad)
        loss_p = F.l1_loss(pred_pitch, tgt_p_pad)
        pf = pred_hubert.permute(0,2,1); tl = tgt_h_pad.permute(0,2,1)
        loss_sdtw = self.sdtw(pf, tl)
        labels = torch.zeros(B, T+1, dtype=torch.long, device=device)
        labels[:,-1] = 1
        loss_ce = F.cross_entropy(logits.view(-1,2), labels.view(-1))
        total = loss_h + loss_p + self.sdtw_weight*loss_sdtw + self.ce_weight*loss_ce
        return total, {'hubert_l1':loss_h, 'pitch_l1':loss_p, 'sdtw':loss_sdtw, 'ce':loss_ce}

    def greedy_decode(self, src_hubert, src_pitch, max_len=200):
        B, S, _ = src_hubert.size()
        device = src_hubert.device
        # Encode source
        src = self.src_hubert_proj(src_hubert) + self.src_pitch_proj(src_pitch.unsqueeze(-1))
        memory = self.encoder(src)

        # Initialize decoder
        current = self.bos_token.expand(B,1,self.d_model)
        decoded_h = []
        decoded_p = []
        ended = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            t_len = current.size(1)
            mask = compute_diagonal_mask(t_len, S, self.nu, device)
            x = current
            for layer in self.decoder_layers:
                x2, _ = layer['self_attn'](x, x, x)
                x = x + x2; x = layer['ffn'](x)
                x2, _ = layer['cross_attn'](x, memory, memory, attn_mask=mask)
                x = x + x2; x = layer['ffn'](x)

            last = x[:, -1, :]
            logits = self.token_classifier(last)
            pred_cls = logits.argmax(dim=-1)

            # Predict features
            h_pred = self.out_hubert(last)         # (B, H_dim)
            p_pred = self.out_pitch(last).squeeze(-1)# (B,)
            decoded_h.append(h_pred.unsqueeze(1))
            decoded_p.append(p_pred.unsqueeze(1))

            # Fuse for next input
            fused_h = self.tgt_hubert_proj(h_pred)
            fused_p = self.tgt_pitch_proj(p_pred.unsqueeze(-1))
            fused = (fused_h + fused_p).unsqueeze(1)
            current = torch.cat([current, fused], dim=1)

            ended = ended | (pred_cls == 1)
            if ended.all():
                break

        hubert_seq = torch.cat(decoded_h, dim=1)
        pitch_seq  = torch.cat(decoded_p, dim=1).squeeze(-1)
        return hubert_seq, pitch_seq