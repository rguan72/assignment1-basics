import math
from collections.abc import Callable
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: torch.nn.Parameter, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
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
                p.data = p.data - lr * weight_decay * p.data
                m = state.get("m", torch.zeros(p.data.shape))
                v = state.get("v", torch.zeros(p.data.shape))
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad ** 2
                p.data = p.data - lr_adj * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["v"] = v
                state["m"] = m
        return loss
                
                
    