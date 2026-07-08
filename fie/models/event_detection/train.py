"""
Training script for the Football Transformer.

Uses PyTorch Lightning for clean training loop, checkpointing, and logging.

Usage:
    # Download data first (one-time):
    python scripts/download_statsbomb.py

    # Precompute fast dataset (one-time, ~15 min for 500 matches):
    python scripts/precompute_dataset.py --max-matches 500

    # Train (fast, GPU-saturated):
    python -m fie.models.event_detection.train \
        --epochs 20 --batch-size 256 --device cuda \
        --fast --num-workers 4 --precision bf16-mixed

    # Train (original, no precompute needed):
    python -m fie.models.event_detection.train \
        --epochs 20 --batch-size 64 --device cuda \
        --max-matches 100 --num-workers 0 --precision bf16-mixed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, random_split

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False

from fie.data.statsbomb import StatsBombEventDataset, StatsBombFastDataset
from fie.models.event_detection.model import FootballTransformer, FootballTransformerConfig

# Allow loading custom classes from checkpoints (PyTorch 2.6+)
torch.serialization.add_safe_globals([FootballTransformerConfig])


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class FootballTransformerLit(L.LightningModule if _HAS_LIGHTNING else object):
    """Lightning wrapper for training."""

    def __init__(
        self,
        config: FootballTransformerConfig,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = FootballTransformer(config)
        self.config = config
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weights = class_weights

    def forward(self, players, masks):
        return self.model(players, masks)

    def _step(self, batch: tuple, stage: str) -> torch.Tensor:
        players, masks, labels = batch
        logits = self(players, masks)
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(logits, labels, weight=weights)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": sched}


# ---------------------------------------------------------------------------
# Pure PyTorch fallback trainer (if Lightning not installed)
# ---------------------------------------------------------------------------

def train_pytorch(
    model: FootballTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    checkpoint_dir: Path,
    class_weights: torch.Tensor | None = None,
) -> None:
    """Minimal training loop without Lightning."""
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = train_acc = 0.0
        for players, masks, labels in train_loader:
            players, masks, labels = players.to(device), masks.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(players, masks)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            train_acc += (logits.argmax(1) == labels).float().mean().item()

        sched.step()
        n = len(train_loader)
        train_loss /= n
        train_acc /= n

        # --- Validate ---
        model.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for players, masks, labels in val_loader:
                players, masks, labels = players.to(device), masks.to(device), labels.to(device)
                logits = model(players, masks)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                val_loss += loss.item()
                val_acc += (logits.argmax(1) == labels).float().mean().item()

        n = len(val_loader)
        val_loss /= n
        val_acc /= n

        logger.info(
            "Epoch {:>3} | train loss={:.4f} acc={:.3f} | val loss={:.4f} acc={:.3f}",
            epoch, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            logger.info("  ✓ Checkpoint saved: {}", ckpt_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SliceDataset — picklable train/val split for Windows multiprocessing
# ---------------------------------------------------------------------------

class _SliceDataset:
    """Sequential slice of a dataset — pickles as 3 small values (no index list)."""
    def __init__(self, dataset, start: int, end: int):
        self.dataset = dataset
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Football Transformer")
    parser.add_argument("--data-dir", default="data/statsbomb")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="bf16-mixed",
                        help="Precision: bf16-mixed (fastest on RTX), 16-mixed, 32")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile() — 10-20%% faster after warmup")
    parser.add_argument("--cache", action="store_true",
                        help="Cache dataset to .pt file (fast reload next run)")
    parser.add_argument("--resume", default=None,
                        help="Path to .ckpt file to resume training")
    parser.add_argument("--fast", action="store_true",
                        help="Use StatsBombFastDataset (memmap). "
                             "Run scripts/precompute_dataset.py first.")
    parser.add_argument("--memmap-dir", default="data/fast_cache",
                        help="Directory with precomputed memmap files (default: data/fast_cache)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: {}", args.device)

    # Tensor Cores (RTX series) — faster matmul
    if args.device == "cuda":
        torch.set_float32_matmul_precision("medium")

    # --- Dataset ---
    if args.fast:
        logger.info("Using StatsBombFastDataset (memmap) from {}", args.memmap_dir)
        dataset = StatsBombFastDataset(
            cache_dir=args.memmap_dir,
            augment=True,
        )
        logger.info("Tip: num-workers >= 4 recommended with --fast")
    else:
        cache_path = Path(args.data_dir) / f"cache_{args.max_matches or 'all'}.pt"
        if args.cache and cache_path.exists():
            logger.info("Loading cached dataset from {}", cache_path)
            dataset = torch.load(cache_path, weights_only=False)
        else:
            dataset = StatsBombEventDataset(
                data_dir=args.data_dir,
                max_matches=args.max_matches,
            )
            if args.cache:
                logger.info("Saving dataset cache to {}", cache_path)
                torch.save(dataset, cache_path)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    if args.fast:
        # Sequential split: pickle is tiny (no 4M index list) → num_workers works on Windows
        train_ds = _SliceDataset(dataset, 0, train_size)
        val_ds   = _SliceDataset(dataset, train_size, len(dataset))
    else:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    logger.info("Train: {} | Val: {} samples", train_size, val_size)

    # --- Class weights (handle imbalance) ---
    if hasattr(dataset, "_labels"):
        # StatsBombFastDataset — labels stored as numpy memmap, fast path
        lbl_arr = np.array(dataset._labels)
        counts = torch.zeros(9)
        for c in range(9):
            counts[c] = int((lbl_arr == c).sum())
    else:
        # StatsBombEventDataset — labels in .samples list
        counts = torch.zeros(9)
        for _, lbl in dataset.samples:
            counts[lbl] += 1
    class_weights = (counts.sum() / (9 * counts.clamp(min=1))).float()
    logger.info("Class weights: {}", class_weights.tolist())

    # --- Model ---
    config = FootballTransformerConfig()
    model = FootballTransformer(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: {:,}", n_params)

    # --- Train ---
    if _HAS_LIGHTNING:
        lit_model = FootballTransformerLit(config, lr=args.lr, class_weights=class_weights)

        # torch.compile — 10-20% faster after warmup (requires PyTorch 2.0+)
        if args.compile:
            try:
                lit_model.model = torch.compile(lit_model.model)
                logger.info("torch.compile() enabled")
            except Exception as e:
                logger.warning("torch.compile() failed: {} — skipping", e)

        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator=args.device,
            devices=1,
            precision=args.precision,
            logger=CSVLogger(str(checkpoint_dir), name="events"),
            callbacks=[
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="best_model",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                ),
                ModelCheckpoint(
                    dirpath=str(checkpoint_dir),
                    filename="last",
                    save_last=True,
                    every_n_epochs=1,
                ),
                EarlyStopping(monitor="val/loss", patience=7, mode="min"),
            ],
            gradient_clip_val=1.0,
            log_every_n_steps=10,
        )
        ckpt_path = args.resume
        if ckpt_path:
            logger.info("Resuming from checkpoint: {}", ckpt_path)
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)
        logger.info("Best checkpoint: {}", trainer.checkpoint_callback.best_model_path)
    else:
        logger.warning("Lightning not installed — using basic training loop")
        train_pytorch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            checkpoint_dir=checkpoint_dir,
            class_weights=class_weights,
        )


if __name__ == "__main__":
    main()
