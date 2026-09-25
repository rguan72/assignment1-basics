"""Train a TransformerLM on pre-tokenized data.

Examples:
    uv run python -m cs336_basics.training_together --iters 200 --no-wandb
    uv run python -m cs336_basics.training_together --modal --iters 5000 --batch-size 128
"""
import math

import argparse
import dataclasses
import logging
import time
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import modal
import numpy as np
import numpy.typing as npt
import torch
import wandb

from cs336_basics import module, optimizer, training

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # data / checkpointing
    train_data: str = "tinystories_train_tokenized.npy"
    valid_data: str = "tinystories_valid_tokenized.npy"
    checkpoint_path: str = "tinystories_checkpoint.pkl"
    # model
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000
    # optimization
    batch_size: int = 64
    iters: int = 100
    max_learning_rate: float = 1e-3
    min_learning_rate: float | None = None  # default: max_learning_rate / 10
    warmup_iters: int | None = None  # default: iters // 20
    cosine_cycle_iters: int | None = None  # default: iters
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_l2_norm: float = 1.0
    # run
    device: str | None = None  # default: cuda > mps > cpu
    seed: int = 0
    log_interval: int = 10
    validation_cycle: int = 10
    val_batches: int = 10
    checkpoint_cycle: int = 100
    wandb: bool = True
    wandb_project: str = "cs336-basics"
    run_name: str | None = None

    def __post_init__(self):
        if self.min_learning_rate is None:
            self.min_learning_rate = self.max_learning_rate / 10
        if self.warmup_iters is None:
            self.warmup_iters = self.iters // 20
        if self.cosine_cycle_iters is None:
            self.cosine_cycle_iters = self.iters
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

def get_validation_batch(x: npt.NDArray[np.uint16], batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 1 ≤ 𝑖 ≤ 𝑛 − 𝑚
    # n: dataset size
    # m: context_length
    # [0, m) and [1, m+1) at the low end
    # [n-1-m, n-1) and [n-m, n) at the high end (why we can sample i between 1 and n-m)
    starting_indices = np.arange(0, batch_size * context_length, context_length).reshape(batch_size, 1)
    vary = np.arange(0, context_length).reshape(1, context_length)
    indices = starting_indices + vary
    return (torch.tensor(x[indices].astype(np.int64), device=device), torch.tensor(x[indices+1].astype(np.int64), device=device)) 


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: np.ndarray, cfg: Config) -> tuple[float, float]:
    losses = []
    for _ in range(cfg.val_batches):
        inputs, targets = get_validation_batch(data, cfg.batch_size, cfg.context_length, cfg.device)
        losses.append(module.cross_entropy(model(inputs), targets).item())
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train(cfg: Config) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        config=dataclasses.asdict(cfg),
        mode="online" if cfg.wandb else "disabled",
    )

    train_data = np.load(cfg.train_data, mmap_mode="r")
    valid_data = np.load(cfg.valid_data, mmap_mode="r")
    logger.info("train tokens: %s, valid tokens: %s", f"{train_data.size:,}", f"{valid_data.size:,}")

    model = module.TransformerLM(
        cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff, cfg.rope_theta
    ).to(cfg.device)
    optim = optimizer.AdamW(
        model.parameters(), cfg.max_learning_rate, cfg.weight_decay, (cfg.beta1, cfg.beta2), cfg.eps
    )
    num_params = sum(p.numel() for p in model.parameters())
    run.summary["num_params"] = num_params
    logger.info("model params: %s, device: %s", f"{num_params:,}", cfg.device)

    t0 = time.perf_counter()
    for it in range(cfg.iters):
        model.zero_grad()
        inputs, targets = training.get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model.forward(inputs)
        loss = module.cross_entropy(logits, targets)
        loss.backward()
        optimizer.gradient_clipping(model.parameters(), cfg.max_l2_norm)
        lr = optimizer.get_lr_cosine_schedule(
            it, cfg.max_learning_rate, cfg.min_learning_rate, cfg.warmup_iters, cfg.cosine_cycle_iters
        )
        for group in optim.param_groups:
            group["lr"] = lr
        optim.step()

        step = it + 1
        last_step = step == cfg.iters

        if step % cfg.log_interval == 0 or last_step:
            train_loss = loss.item()
            elapsed = time.perf_counter() - t0
            run.log({"train/loss": train_loss, "train/lr": lr}, step=step)
            logger.info("step %d/%d | train loss %.4f | lr %.2e | %.1fs", step, cfg.iters, train_loss, lr, elapsed)

        if step % cfg.validation_cycle == 0 or last_step:
            val_loss, val_perplexity = evaluate(model, valid_data, cfg)
            run.log({"val/loss": val_loss}, step=step)
            logger.info("step %d/%d | val loss %.4f", step, cfg.iters, val_loss)
            run.log({"val/perplexity": val_perplexity}, step=step)
            logger.info("step %d/%d | val perplexity %.4f", step, cfg.iters, val_perplexity)

        if step % cfg.checkpoint_cycle == 0 or last_step:
            Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            training.save_checkpoint(model, optim, step, cfg.checkpoint_path)
            logger.info("saved checkpoint to %s", cfg.checkpoint_path)

    run.finish()


# ---------------------------------------------------------------------------
# Modal: `--modal` runs `train` remotely on a B200. Data files are uploaded once
# to the `cs336-data` volume; checkpoints are written back to it.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
VOLUME_MOUNT = "/data"

app = modal.App("cs336-basics-train")
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_sync(str(REPO_ROOT)).add_local_python_source("cs336_basics")


@app.function(
    image=image,
    gpu="B200",
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("wandb-secret-2")],
    timeout=24 * 60 * 60,
)
def train_remote(cfg_dict: dict) -> None:
    setup_logging()
    cfg = Config(**cfg_dict)
    train(cfg)
    volume.commit()


def run_on_modal(cfg: Config) -> None:
    existing = {entry.path for entry in volume.listdir("/")}
    to_upload = [Path(p) for p in (cfg.train_data, cfg.valid_data) if Path(p).name not in existing]
    if to_upload:
        logger.info("uploading %s to modal volume", [p.name for p in to_upload])
        with volume.batch_upload() as batch:
            for path in to_upload:
                batch.put_file(path, f"/{path.name}")

    remote_cfg = dataclasses.replace(
        cfg,
        train_data=f"{VOLUME_MOUNT}/{Path(cfg.train_data).name}",
        valid_data=f"{VOLUME_MOUNT}/{Path(cfg.valid_data).name}",
        checkpoint_path=f"{VOLUME_MOUNT}/checkpoints/{Path(cfg.checkpoint_path).name}",
        device="cuda",
    )
    with modal.enable_output(), app.run():
        train_remote.remote(dataclasses.asdict(remote_cfg))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> tuple[Config, argparse.Namespace]:
    """Every Config field becomes a `--flag`; defaults come from the dataclass."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modal", action="store_true", help="run on Modal with a B200 GPU")
    hints = typing.get_type_hints(Config)
    for field in dataclasses.fields(Config):
        flag = "--" + field.name.replace("_", "-")
        typ = hints[field.name]
        if typ is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=field.default)
            continue
        if isinstance(typ, types.UnionType):  # `X | None`
            typ = next(t for t in typing.get_args(typ) if t is not types.NoneType)
        parser.add_argument(flag, type=typ, default=field.default, help=f"(default: {field.default})")
    args = parser.parse_args()
    cfg_kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(Config)}
    if args.modal and cfg_kwargs["device"] is None:
        cfg_kwargs["device"] = "cuda"  # don't resolve against the local machine
    return Config(**cfg_kwargs), args


def main() -> None:
    setup_logging()
    cfg, args = parse_args()
    if args.modal:
        run_on_modal(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
