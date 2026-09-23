import einops
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        std: float = (2 / (in_features + out_features))**0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        # params: in x out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: prod(b_dims) x in
        # weight: in x out
        # FLOPS: 2*prod(b_dims)(in)(out)
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        # params: num_embeddings x embedding_dim
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMS(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        # params: d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(einops.reduce(x ** 2, 'b t c -> b t 1', 'mean') + self.eps)
        result = (x * self.weight) / rms
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        # params: 3 * d_model * d_ff
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x) # FLOPS: 2 x prod(b_dims) x d_model x d_ff
        silu = w1x * torch.sigmoid(w1x)
        w3x = self.w3(x)
        return self.w2(silu * w3x)
        # total FLOPS: 6 x prod(b_dims) x d_model x d_ff

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
    # FLOPS: 2 * prod(b_dims) * q * dk * k
    if mask is not None:
        dot_product.masked_fill_(~mask, float('-inf'))
    probabilities = softmax(dot_product, -1)
    # FLOPS: 2 * prod(b_dims) * q * k * dv
    return einops.einsum(probabilities, V, '... q k, ... k dv -> ... q dv')
    # Total FLOPS: 2 * prod(b_dims) * q * k * (dk + dv)

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
        # params: 4 * num_heads * dk * d_model

    def forward(self, x: Float[Tensor, "batch time channel"], token_positions: Int[Tensor, "... sequence_length"] | None) -> Tensor:
        t = x.shape[1]
        if token_positions is None:
            token_positions = torch.arange(t)
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        # FLOPS: 6*b*t*num_heads*dk*d_model
        q_batched = einops.rearrange(q, 'b t (n k) -> b n t k', k=self.dk, n=self.num_heads)
        q_batched_rope = self.rope.forward(q_batched, token_positions)
        k_batched = einops.rearrange(k, 'b t (n k) -> b n t k', k=self.dk, n=self.num_heads)
        k_batched_rope = self.rope.forward(k_batched, token_positions)
        v_batched = einops.rearrange(v, 'b t (n k) -> b n t k', k=self.dk, n=self.num_heads)
        mask = torch.tril(torch.ones(t, t)) == 1
        attn = scaled_dot_product_attention(q_batched_rope, k_batched_rope, v_batched, mask) # b n t dk
        # FLOPS: 2 * (b * num_heads) * t * t * (dk + dk) = 4(b*num_heads)(t^2)(dk)
        return self.output_proj.forward(einops.rearrange(attn, 'b n t dk -> b t (n dk)'))
        # FLOPS: 2(b*t)(num_heads*dk)(d_model)
        # Total FLOPS: (2*b*t*num_heads*dk)(d_model + 2t + 3d_model) = 4(b*t*num_heads*dk)(2d_model+t)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.ln1 = RMS(d_model) # params: d_model
        self.ln2 = RMS(d_model) # params: d_model
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len) # params: 4 * num_heads * dk * d_model
        self.ffn = SwiGLU(d_model, d_ff) # params: 3 * d_model * d_ff
        # Total params: 2d_model + 4d_model(num_heads*dk) + 3d_model * d_ff = d_model(4num_heads*dk + 3d_ff + 2) 

    def forward(self, x: Float[Tensor, "batch time channel"]) -> Tensor:
        y = x + self.attn.forward(self.ln1.forward(x), None) # FLOPS: 4(b*t*num_heads*(d_model//num_heads))(2d_model+t)
        # assume d_model % num_heads == 0
        # FLOPS (simplified): 4(b*t*d_model)(2d_model+t)
        y = y + self.ffn.forward(self.ln2.forward(y)) # FLOPS: 6(b*t*d_model*d_ff)
        return y
        # Total FLOPS: 4(b*max_seq_len*d_model)(2d_model+max_seq_len) + 6(b*max_seq_len*d_model*d_ff) 
        # = 2(b*max_seq_len*d_model)(2(2d_model+max_seq_len) + 3d_ff) 
        # = 2(b*max_seq_len*d_model)(4d_model + 2max_seq_len + 3d_ff)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model) # params: vocab_size * d_model
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)])
        # params: num_layers * d_model(4num_heads*dk + 3d_ff + 2)
        self.ln_final = RMS(d_model) # d_model
        self.lm_head = Linear(d_model, vocab_size) # d_model * vocab_size
        # total params: vocab_size * d_model + num_layers * d_model(4num_heads*dk + 3d_ff + 2) + d_model + d_model * vocab_size
        # = d_model(2vocab_size + num_layers(4num_heads*dk + 3d_ff + 2) + 1)
        # = d_model(2vocab_size + num_layers(4d_model + 3d_ff + 2) + 1)
        # assuming d_ff = 8/3 * d_model
        # = d_model(2vocab_size + num_layers(12d_model + 2) + 1)

    def forward(self, x: Int[Tensor, "batch time"]) -> Tensor:
        x= self.token_embeddings.forward(x)
        x = self.layers.forward(x) # FLOPS: num_layers * 2(b*context_length*d_model)(4d_model + 2context_length + 3d_ff)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x) # FLOPS: 2*b*context_length*(d_model)(vocab_size)
        return x
        # total FLOPS: num_layers * 2(b*context_length*d_model)(4d_model + 2context_length + 3d_ff) + 2*b*context_length*(d_model)(vocab_size)
        # = 2b*context_length*d_model(num_layers*(4d_model + 2context_length + 3d_ff) + vocab_size)

def cross_entropy(logits: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    logits_stable = logits - torch.amax(logits, dim=-1, keepdim=True)
    lhs = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    rhs = torch.gather(input=logits_stable, dim=-1, index=targets.unsqueeze(-1)).squeeze()
    return torch.mean(lhs - rhs)