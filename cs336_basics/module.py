from torch import nn
import torch
import einops

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(out_features, in_features))
        std: float = (2 / in_features + out_features)**0.5
        nn.init.trunc_normal_(self.w, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.w, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]

class RMS(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.w = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # B x T x C
        rms = torch.sqrt(einops.reduce(x ** 2, 'b t c -> b t 1', 'mean') + self.eps)
        result = (x * self.w) / rms
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.linear3 = Linear(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.linear1(x) # B x T x d_ff
        silu = w1x * torch.sigmoid(w1x) # B x T x d_ff
        w3x = self.linear3(x) # B x T x d_ff
        return self.linear2(silu * w3x)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        k: int =  d_k // 2
        positions = torch.arange(max_seq_len, device=device)
        frequencies = 1 / torch.pow(theta, (2 * torch.arange(start=1, end=k+1, device=device) - 2) / d_k)
        angle_table = torch.outer(positions, frequencies)
        self.register_buffer(name="sin_table", tensor=torch.sin(angle_table), persistent=False)
        self.register_buffer(name="cos_table", tensor=torch.cos(angle_table), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: B x T x C
        cosines = self.cos_table[token_positions] # TxC//2
        repeated_cosines = einops.rearrange([cosines, cosines], 'two t k -> t (k two)')
        odds = x[:, :, 1::2] # BxTxC/2
        evens= x[:, :, ::2] # BxTxC/2
        merged = einops.rearrange([-odds, evens], 'two b t c -> b t (c two)')
        sines = self.sin_table[token_positions]
        repeated_sines = einops.rearrange([sines, sines], 'two t k -> t (k two)')
        return merged * repeated_sines + x * repeated_cosines
