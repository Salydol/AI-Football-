"""
Precompute StatsBomb dataset into fast numpy memmap files.

Run this ONCE before training with --fast flag.
Converts raw JSON events → compact binary files (~1 GB for 500 matches).
After precomputation, training with --fast --num-workers 4 is 10-15x faster.

Usage:
    # All downloaded matches:
    python scripts/precompute_dataset.py

    # Limit to N matches:
    python scripts/precompute_dataset.py --max-matches 500

    # Custom paths:
    python scripts/precompute_dataset.py \
        --data-dir data/statsbomb \
        --cache-dir data/fast_cache \
        --max-matches 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from loguru import logger

from fie.data.statsbomb import StatsBombFastDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute StatsBomb dataset → numpy memmap"
    )
    parser.add_argument("--data-dir", default="data/statsbomb",
                        help="StatsBomb events directory (default: data/statsbomb)")
    parser.add_argument("--cache-dir", default="data/fast_cache",
                        help="Output directory for memmap files (default: data/fast_cache)")
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Limit number of matches (default: all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    if not (data_dir / "events").exists():
        logger.error(
            "StatsBomb events not found at {}. "
            "Run: python scripts/download_statsbomb.py",
            data_dir / "events",
        )
        return

    logger.info("=" * 60)
    logger.info("StatsBomb Fast Dataset Precomputation")
    logger.info("  Source:     {}", data_dir)
    logger.info("  Output:     {}", cache_dir)
    logger.info("  Max matches: {}", args.max_matches or "all")
    logger.info("=" * 60)
    logger.info("This takes 10-20 minutes for 500 matches.")
    logger.info("After this, training with --fast is 10-15x faster.")

    t0 = time.time()
    ds = StatsBombFastDataset.precompute(
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_matches=args.max_matches,
    )
    elapsed = time.time() - t0

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done! {} samples in {:.1f} minutes", len(ds), elapsed / 60)
    logger.info("")
    logger.info("Cache size:")
    total_mb = sum(
        f.stat().st_size for f in cache_dir.iterdir() if f.is_file()
    ) / 1024 / 1024
    logger.info("  {} MB in {}", int(total_mb), cache_dir)
    logger.info("")
    logger.info("Now train with:")
    logger.info(
        "  python -m fie.models.event_detection.train "
        "--epochs 20 --batch-size 256 --device cuda "
        "--fast --num-workers 4 --precision bf16-mixed"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
