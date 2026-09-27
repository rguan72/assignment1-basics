import torch
from cs336_basics import tokenizer

context_length = 256

@torch.no_grad()
def decode(model: torch.nn.Module, prompt_tokens: list[int], eos: int = 256, max_tokens: int = 256, temperature: float = 1.0, top_p: float | None = None) -> list[int]:
    # one forward pass
    input_token_length = len(prompt_tokens)
    input_tokens = torch.tensor(prompt_tokens, device=next(model.parameters()).device).unsqueeze(0)
    for _ in range(max_tokens):
        logits: torch.Tensor = model.forward(input_tokens)[:, -1, :] # b x vocab
        logits *= 1.0 / temperature
        # softmax   
        logits -= torch.amax(logits, dim=-1, keepdim=True) # b x vocab
        logits = torch.exp(logits)
        probs = logits / torch.sum(logits, dim=-1, keepdim=True)
        if top_p:
            sorted_probs, indices = torch.sort(probs, descending=True) # b x vocab
            # want: cumsums[j] is prob mass for sortedprobes[j:]
            # then: mask = cumsums < 1-top_p
            # [0.6, 0.3, 0.05, 0.05], top_p = 0.7
            # cumsum = [0.6, 0.9, 0.95, 1.0]
            # 1 - cumsum with last shifted to first
            # [1.0, 0.4, 0.1, 0.05]
            leftover_prob_mass = torch.cat([torch.ones((sorted_probs.shape[0], 1), device=sorted_probs.device), 1 - torch.cumsum(sorted_probs, dim=-1)[..., :-1]], dim=-1)
            mask = leftover_prob_mass < 1 - top_p
            sorted_probs = torch.masked_fill(sorted_probs, mask, 0.0)
            sorted_probs *= 1 / torch.sum(sorted_probs, dim=-1, keepdim=True)
            probs = torch.empty_like(sorted_probs).scatter_(-1, indices, sorted_probs)
        next_tokens = torch.multinomial(probs, num_samples=1) # b x 1
        input_tokens = torch.cat([input_tokens, next_tokens], dim=-1)
        if next_tokens.item() == eos:
            break
        if len(input_tokens) >= context_length:
            break
    return input_tokens.squeeze().tolist()[input_token_length:]

def decode_str(input: str, model: torch.nn.Module, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] = ["<|endoftext|>"], eos: int = 256, max_tokens: int = 256, temperature: float = 1.0, top_p: float | None = None) -> str:
    toknizer = tokenizer.Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    input_tokens = toknizer.encode(input)
    output_tokens = decode(model, input_tokens, eos, max_tokens, temperature, top_p)
    return toknizer.decode(output_tokens)
    