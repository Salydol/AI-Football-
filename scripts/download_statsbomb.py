"""
Download StatsBomb Open Data for training the Event Detection model.

Usage:
    # Download all ~900 matches (takes ~10 min):
    python scripts/download_statsbomb.py

    # Download only 50 matches (quick test):
    python scripts/download_statsbomb.py --max-matches 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fie.data.statsbomb import StatsBombDownloader

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="data/statsbomb")
parser.add_argument("--max-matches", type=int, default=None)
args = parser.parse_args()

dl = StatsBombDownloader(data_dir=args.data_dir)
dl.download(max_matches=args.max_matches)
print(f"\nDone. Data saved to: {Path(args.data_dir).resolve()}")
