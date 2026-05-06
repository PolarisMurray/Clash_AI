"""Offline-RL / behavior-cloning training loop for the Perceiver-DT agent.

Usage (smoke test, synthetic data):
    python -m training.train_policy --smoke

Usage (real .npz trajectories under outputs/replay_dataset/):
    python -m training.train_policy --data outputs/replay_dataset --epochs 10
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from configs.policy_config import FullConfig, default_config
from data.replay_dataset import (
    Trajectory,
    build_dataloader,
    build_random_trajectory,
    discover_trajectories,
)
from models.policy.agent import ClashRoyaleAgent
from models.policy.policy_heads import policy_loss


# ------------------------- device ------------------------- #

def select_device(pref: str) -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


# ------------------------- optim helpers ------------------------- #

def cosine_lr_with_warmup(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def split_decay_params(model: torch.nn.Module):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name.lower() or "norm" in name.lower() or "bias" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


# ------------------------- logging ------------------------- #

class Logger:
    def __init__(self, cfg: FullConfig, run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.tb = None
        self.wb = None
        if cfg.train.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(run_dir / "tb"))
            except Exception as exc:                                          # noqa: BLE001
                print(f"[logger] tensorboard disabled: {exc}")
        if cfg.train.use_wandb:
            try:
                import wandb
                self.wb = wandb.init(project=cfg.train.wandb_project, dir=str(run_dir), config=cfg.to_dict())
            except Exception as exc:                                          # noqa: BLE001
                print(f"[logger] wandb disabled: {exc}")

    def log(self, metrics: dict, step: int, prefix: str = "train"):
        if self.tb is not None:
            for k, v in metrics.items():
                self.tb.add_scalar(f"{prefix}/{k}", float(v), step)
        if self.wb is not None:
            self.wb.log({f"{prefix}/{k}": float(v) for k, v in metrics.items()}, step=step)

    def close(self):
        if self.tb is not None:
            self.tb.close()


# ------------------------- training step ------------------------- #

def train_one_epoch(
    agent: ClashRoyaleAgent,
    loader,
    optimizer,
    scheduler,
    cfg: FullConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    logger: Logger,
) -> int:
    agent.train()
    loss_w = {
        "select": cfg.model.heads.loss_select_w,
        "pos": cfg.model.heads.loss_pos_w,
        "delay": cfg.model.heads.loss_delay_w,
    }
    grad_accum = cfg.train.grad_accum
    optimizer.zero_grad(set_to_none=True)
    for i, (batch, targets) in enumerate(loader):
        batch = batch.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        action_mask = targets.pop("action_mask")

        out = agent(batch)
        metrics = policy_loss(out, targets, action_mask, loss_w)
        loss = metrics["loss"] / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0:
            if cfg.train.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg.train.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                logger.log(
                    {
                        "loss": metrics["loss"].item(),
                        "loss_select": metrics["loss_select"].item(),
                        "loss_pos": metrics["loss_pos"].item(),
                        "loss_delay": metrics["loss_delay"].item(),
                        "acc_select": metrics["acc_select"].item(),
                        "acc_pos": metrics["acc_pos"].item(),
                        "acc_delay": metrics["acc_delay"].item(),
                        "lr": lr,
                        "epoch": epoch + i / max(1, len(loader)),
                    },
                    step=global_step,
                )
                print(
                    f"epoch {epoch} step {global_step}  loss={metrics['loss'].item():.4f} "
                    f"sel={metrics['loss_select'].item():.3f} pos={metrics['loss_pos'].item():.3f} "
                    f"del={metrics['loss_delay'].item():.3f} lr={lr:.2e}"
                )
    return global_step


def save_checkpoint(agent: ClashRoyaleAgent, cfg: FullConfig, run_dir: Path, epoch: int):
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save({"epoch": epoch, "state_dict": agent.state_dict(), "config": cfg.to_dict()}, path)
    return path


# ------------------------- main ------------------------- #

def load_or_synth(data_root: Optional[Path], cfg: FullConfig, smoke: bool):
    if smoke or data_root is None:
        print("[data] using synthetic trajectories (smoke test mode)")
        return [build_random_trajectory(cfg, 64, seed=i) for i in range(8)]
    files = discover_trajectories(data_root)
    if not files:
        raise SystemExit(f"No .npz trajectory files found under {data_root}")
    print(f"[data] loading {len(files)} trajectories from {data_root}")
    return [Trajectory.from_npz(p) for p in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None, help="Trajectory dir (.npz files)")
    parser.add_argument("--out", type=Path, default=None, help="Override output dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use synthetic data and tiny model for a smoke test")
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None: overrides["epochs"] = args.epochs
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.seq_len is not None:
        overrides["seq_len"] = args.seq_len
    if args.lr is not None: overrides["lr"] = args.lr
    if args.device is not None: overrides["device"] = args.device
    if args.num_workers is not None: overrides["num_workers"] = args.num_workers
    if args.smoke:
        overrides.update(dict(
            batch_size=2, epochs=1, seq_len=4, num_workers=0, log_interval=1,
        ))

    cfg = default_config(**overrides)

    if args.smoke:
        cfg.model.perceiver.n_self_layers = 1
        cfg.model.perceiver.n_cross_blocks = 1
        cfg.model.n_layers = 2
        cfg.model.perceiver.n_latents = 16
        cfg.tokenizer.image_size = (64, 64)
        cfg.tokenizer.patch_size = 16
        cfg.tokenizer.max_objects = 8

    run_dir = Path(args.out) if args.out is not None else cfg.train.out_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output dir: {run_dir}")

    device = select_device(cfg.train.device)
    print(f"[run] device: {device}")
    torch.manual_seed(cfg.train.seed)

    trajs = load_or_synth(args.data, cfg, smoke=args.smoke)
    train_loader = build_dataloader(cfg, trajs, split="train")
    print(f"[data] {sum(len(t) for t in trajs)} frames, {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")

    agent = ClashRoyaleAgent(cfg).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[model] params: {n_params/1e6:.2f}M")

    decay, no_decay = split_decay_params(agent)
    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
        betas=cfg.train.betas,
    )
    total_steps = max(1, cfg.train.epochs * (len(train_loader) // max(1, cfg.train.grad_accum)))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_with_warmup(s, cfg.train.warmup_steps, total_steps),
    )

    logger = Logger(cfg, run_dir)
    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        global_step = train_one_epoch(
            agent, train_loader, optimizer, scheduler, cfg, device,
            epoch=epoch, global_step=global_step, logger=logger,
        )
        if (epoch + 1) % cfg.train.ckpt_interval_epochs == 0:
            path = save_checkpoint(agent, cfg, run_dir, epoch)
            print(f"[ckpt] saved {path}")
    logger.close()
    print(f"[run] done in {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
