from torch import nn
import torch
import einops
from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        std: float = (2 / (in_features + out_features))**0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMS(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # B x T x C
        rms = torch.sqrt(einops.reduce(x ** 2, 'b t c -> b t 1', 'mean') + self.eps)
        result = (x * self.weight) / rms
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x) # B x T x d_ff
        silu = w1x * torch.sigmoid(w1x) # B x T x d_ff
        w3x = self.w3(x) # B x T x d_ff
        return self.w2(silu * w3x)

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
        repeated_cosines = einops.rearrange([cosines, cosines], 'two ... k -> ... (k two)')
        odds = x[..., 1::2] # BxTxC/2
        evens= x[..., ::2] # BxTxC/2
        merged = einops.rearrange([-odds, evens], 'two ... c -> ... (c two)')
        sines = self.sin_table[token_positions]
        repeated_sines = einops.rearrange([sines, sines], 'two ... k -> ... (k two)')
        return merged * repeated_sines + x * repeated_cosines

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    stabilized = x - torch.amax(x, dim=i, keepdim=True)
    exps = torch.exp(stabilized)
    return exps / torch.sum(exps, dim=i, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... keys d_v"]:
    d_k = Q.shape[-1]
    dot_product: Tensor = einops.einsum(Q, K, '... q dk, ... k dk -> ... q k') / (d_k**0.5)
    if mask is not None:
        dot_product.masked_fill_(~mask, float('-inf'))
    probabilities = softmax(dot_product, -1)
    return einops.einsum(probabilities, V, '... q k, ... k dv -> ... q dv')

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.output_proj = Linear(num_heads * self.dk, d_model)
        self.q_proj = Linear(d_model, num_heads * self.dk)
        self.k_proj = Linear(d_model, num_heads * self.dk)
        self.v_proj = Linear(d_model, num_heads * self.dk)
        self.rope = RotaryPositionEmbedding(theta, self.dk, max_seq_len)

    def forward(self, x: Float[Tensor, "batch time channel"], token_positions: Int[Tensor, "... sequence_length"] | None) -> Tensor:
        t = x.shape[1]
        if token_positions is None:
            token_positions = torch.arange(t)
        # rope_x = self.rope.forward(x, token_positions)
        q = self.q_proj.forward(x) # b x t x num_heads*dk
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        q_batched = einops.rearrange(q, 'b t (n k) -> b n t k', k=self.dk, n=self.num_heads)
        q_batched_rope = self.rope.forward(q_batched, token_positions)
        k_batched = einops.rearrange(k, 'b t (n k) -> b n t k', k=self.dk, n=self.num_heads)
        k_batched_rope = self.rope.forward(k_batched, token_positions)
        v_batched = einops.rearrange(v, 'b t (n k) -> b n t k', k=self.dk, n=self.num_heads)
        mask = torch.tril(torch.ones(t, t)) == 1
        attn = scaled_dot_product_attention(q_batched_rope, k_batched_rope, v_batched, mask) # b n t dk
        return self.output_proj.forward(einops.rearrange(attn, 'b n t dk -> b t (n dk)'))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.ln1 = RMS(d_model)
        self.ln2 = RMS(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, "batch time channel"]) -> Tensor:
        y = x + self.attn.forward(self.ln1.forward(x), None)
        y = y + self.ffn.forward(self.ln2.forward(y))
        return y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)])
        self.ln_final = RMS(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "batch time"]) -> Tensor:
        x= self.token_embeddings.forward(x)
        x = self.layers.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x

def cross_entropy(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    logits_stable = logits - torch.amax(logits, dim=-1, keepdim=True)
    lhs = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    rhs = torch.gather(input=logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze()
    return torch.mean(lhs - rhs)