import math
from collections.abc import Callable, Iterable
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        return super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            b1 = betas[0]
            b2 = betas[1]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                lr_adj = lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)
                p.data *= (1 - lr * weight_decay)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad ** 2
                p.data -= lr_adj * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["v"] = v
                state["m"] = m
        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 1 / 2 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    squarednorm = 0
    for grad in grads:
        squarednorm += torch.sum(grad ** 2)
    norm = math.sqrt(squarednorm)
    if norm < max_l2_norm:
        return
    scale_factor = max_l2_norm / (norm + 1e-6)
    for grad in grads:
        grad.mul_(scale_factor)
    