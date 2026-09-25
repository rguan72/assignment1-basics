import typing
import os
import numpy as np
import numpy.typing as npt
import torch

def get_batch(x: npt.NDArray[np.uint16], batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 1 ≤ 𝑖 ≤ 𝑛 − 𝑚
    # n: dataset size
    # m: context_length
    # [0, m) and [1, m+1) at the low end
    # [n-1-m, n-1) and [n-m, n) at the high end (why we can sample i between 1 and n-m)
    n = x.size
    starting_indices = np.random.randint(low=0, high=n-context_length, size=batch_size).reshape(batch_size, 1)
    vary = np.arange(0, context_length).reshape(1, context_length)
    indices = starting_indices + vary
    return (torch.tensor(x[indices].astype(np.int64), device=device), torch.tensor(x[indices+1].astype(np.int64), device=device)) 

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    checkpoint_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model_state"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
    return checkpoint_dict["iteration"]