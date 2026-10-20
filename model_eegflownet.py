import math
import torch
import torch.nn as nn

_SIGNAL_LEN: int = 512
MODEL_ARCH = {
    "EOG": dict(feats=64,  n_heads=1, d_k=64, d_v=64, d_ff=512),
}

class Conv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t    = t.reshape(-1)
        half = self.dim // 2
        decay = math.log(10_000) / max(half - 1, 1)
        freqs = torch.exp(
            torch.arange(half, dtype=t.dtype, device=t.device) * -decay
        )
        angles = t[:, None] * freqs[None, :]
        embed  = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return embed.unsqueeze(-1)   # [B, dim, 1]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = d_k ** 0.5

    def forward(self, Q, K, V):
        return torch.matmul(
            (torch.matmul(Q, K.transpose(-1, -2)) / self.scale).softmax(dim=-1), V
        )


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k, self.d_v = d_k, d_v
        self.W_Q  = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K  = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V  = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc   = nn.Linear(d_v * n_heads, d_model,  bias=False)
        self.ln   = nn.LayerNorm(d_model)
        self.attn = ScaledDotProductAttention(d_k)

    def forward(self, q, k, v):
        B, residual = q.size(0), q
        Q = self.W_Q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(v).view(B, -1, self.n_heads, self.d_v).transpose(1, 2)
        ctx = self.attn(Q, K, V).transpose(1, 2).reshape(B, -1, self.n_heads * self.d_v)
        return self.ln(self.fc(ctx) + residual)


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff,    bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,    d_model, bias=False),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.fc(x) + x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn       = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_ffn(self.enc_self_attn(x, x, x))


class MultiScaleConvFront(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scales=(3, 7, 15)):
        super().__init__()
        n       = len(scales)
        base_ch = out_ch // n
        branch_ch = [base_ch] * (n - 1) + [out_ch - base_ch * (n - 1)]

        self.branches = nn.ModuleList([
            nn.Sequential(Conv1d(in_ch, ch, k, padding=k // 2), nn.GELU())
            for k, ch in zip(scales, branch_ch)
        ])
        self.mix = Conv1d(out_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([b(x) for b in self.branches], dim=1))


class CrossAttentionFusion(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.cross = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.ffn   = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.cross(x, context, context))


class JointFiLM(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc_gamma = nn.Linear(2, input_dim)
        self.fc_beta  = nn.Linear(2, input_dim)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, cond_global: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([t_embed, cond_global], dim=-1)   # [B, feats, 2]
        return self.fc_gamma(joint) * x + self.fc_beta(joint)



class EEGFlowNet(nn.Module):

    def __init__(self, feats: int, n_heads: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        d_model  = _SIGNAL_LEN   
        n_levels = 3
        self.front_x    = MultiScaleConvFront(1, feats)
        self.front_cond = MultiScaleConvFront(1, feats)
        self.embed = SinusoidalTimeEmbedding(feats)
        self.enc_x    = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_levels)
        ])
        self.enc_cond = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_levels)
        ])
        self.cross_fuse = nn.ModuleList([
            CrossAttentionFusion(d_model, n_heads, d_k, d_v, d_ff)
            for _ in range(n_levels)
        ])
        self.film = nn.ModuleList([JointFiLM(d_model) for _ in range(n_levels)])
        self.conv_out = nn.Sequential(
            Conv1d(feats, feats, 3, padding=1),
            Conv1d(feats, 1,    3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x    = self.front_x(x)       # [B, feats, 512]
        cond = self.front_cond(cond)  # [B, feats, 512]
        t_embed = self.embed(t)       # [B, feats, 1]

        for enc_x, enc_cond, cross, film in zip(
            self.enc_x, self.enc_cond, self.cross_fuse, self.film
        ):
            cond = enc_cond(cond)
            x    = enc_x(x)
            x    = cross(x, cond)
            cond_pool = cond.mean(dim=-1, keepdim=True)   
            x    = film(x, t_embed, cond_pool)

        return self.conv_out(x)



if __name__ == "__main__":
    x_in = torch.randn(4, 1, 512)
    t_in = torch.rand(4)
    print("EEGFlowNet parameter counts:")
    for noise_type, cfg in MODEL_ARCH.items():
        net = EEGFlowNet(**cfg)
        n_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
        out = net(x_in, x_in, t_in)
        print(f"  {noise_type}: {n_p:>12,} params  output={out.shape}")
