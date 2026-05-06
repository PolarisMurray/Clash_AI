"""Config classes for the simplified Clash Royale policy pipeline.

A single source of truth shared across:
- perception (YOLO + tokenizer)
- model (Perceiver encoder + Decision Transformer + policy heads)
- data (offline replay buffer / dataset)
- training loop

Plain dataclasses — no JAX / Flax dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class TokenizerConfig:
    """Per-frame multimodal tokenization."""

    image_size: Tuple[int, int] = (224, 224)         # (H, W) input to the patchifier
    patch_size: int = 16                              # ViT-style patch size
    n_classes: int = 160                              # YOLO class vocab (155 + small headroom)
    max_objects: int = 32                             # cap detections per frame
    n_cards: int = 128                                # card name vocabulary
    n_hand: int = 5                                   # 4 cards + 1 next-card
    n_elixir: int = 11                                # 0..10 (-1 → unknown bucketed to 0)
    max_delay: int = 20                               # delay head bin count - 1
    grid_rows: int = 36                               # deployment grid rows  (user spec)
    grid_cols: int = 64                               # deployment grid cols  (user spec)
    use_track_id: bool = True
    max_track_ids: int = 64                           # 0 reserved for "no track"

    @property
    def grid_size(self) -> int:
        return self.grid_rows * self.grid_cols


@dataclass
class PerceiverConfig:
    """Perceiver IO latent bottleneck encoder."""

    d_model: int = 256                                # latent dim
    n_latents: int = 64                               # fixed latent array size
    n_cross_heads: int = 4
    n_self_heads: int = 8
    n_self_layers: int = 4                            # self-attn layers per cross-attn block
    n_cross_blocks: int = 2                           # cross→self repetitions (Perceiver IO)
    ff_mult: int = 4
    dropout: float = 0.1
    share_weights_across_blocks: bool = True          # repeat one block, like Perceiver IO


@dataclass
class PolicyHeadsConfig:
    """Output heads for the policy."""

    enable_delay: bool = True
    pos_mode: str = "grid"                            # "grid" (categorical) or "xy" (regression)
    loss_select_w: float = 1.0
    loss_pos_w: float = 1.0
    loss_delay_w: float = 0.5


@dataclass
class ModelConfig:
    """Decision-Transformer over Perceiver latents + scalar tokens."""

    d_model: int = 256                                # must match PerceiverConfig.d_model
    n_heads: int = 8
    n_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    seq_len: int = 30                                 # number of timesteps per sample
    max_timesteps: int = 4096                         # absolute timestep embedding
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    heads: PolicyHeadsConfig = field(default_factory=PolicyHeadsConfig)


@dataclass
class DataConfig:
    """Offline replay dataset & sampling."""

    root: Path = Path("outputs/replay_dataset")
    fps: int = 10
    seq_len: int = 30                                 # match ModelConfig.seq_len
    rtg_scale: float = 1.0                            # divide RTG before feeding model
    weighted_sampling: bool = True                    # upweight action frames
    action_focus_window: int = 10                     # spread action weight to nearby frames
    min_action_frac: float = 0.3                      # fraction of batches forced to be action-anchored
    lr_flip: bool = True                              # left/right augmentation
    card_shuffle: bool = True                         # permute hand-card slots
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Training loop hyper-parameters."""

    seed: int = 42
    batch_size: int = 16
    grad_accum: int = 1
    epochs: int = 10
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    clip_grad_norm: float = 1.0
    log_interval: int = 50
    ckpt_interval_epochs: int = 1
    out_dir: Path = Path("outputs/policy_runs/perceiver_dt")
    device: str = "auto"                              # "auto" | "cuda" | "mps" | "cpu"
    use_wandb: bool = False
    wandb_project: str = "clash-royale-policy"
    use_tensorboard: bool = True


@dataclass
class FullConfig:
    """Convenience bundle so a training script only takes one object."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        assert self.model.seq_len == self.data.seq_len, (
            f"model.seq_len ({self.model.seq_len}) must equal data.seq_len ({self.data.seq_len})"
        )
        assert self.model.d_model == self.model.perceiver.d_model, (
            "model.d_model must equal perceiver.d_model"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def default_config(**overrides) -> FullConfig:
    """Build a FullConfig and apply flat overrides like ``batch_size=8``.

    Overrides are routed by name to whichever sub-config defines them.
    Unknown keys raise.
    """
    cfg = FullConfig()
    routes = {
        "tokenizer": cfg.tokenizer,
        "model": cfg.model,
        "data": cfg.data,
        "train": cfg.train,
        "perceiver": cfg.model.perceiver,
        "heads": cfg.model.heads,
    }
    for key, val in overrides.items():
        placed = False
        # Set on EVERY sub-config that defines this key — useful for shared names
        # like `seq_len` which appears on both model and data configs.
        for sub in routes.values():
            if hasattr(sub, key):
                setattr(sub, key, val)
                placed = True
        if not placed:
            raise KeyError(f"Unknown config key: {key}")
    cfg.__post_init__()
    return cfg


if __name__ == "__main__":
    import json
    cfg = default_config()
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
