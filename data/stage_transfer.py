"""Download the LeJEPA-8 transfer datasets once (train and test splits) under ``--root/<dataset>``,
the layout ``probe.py --transfer-root`` reads.

Usage:
    python -m data.stage_transfer --root $EBJEPA_DSETS/transfer
"""

import argparse
import os

from data.datasets import TRANSFER_DATASETS, build_transfer_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default=",".join(TRANSFER_DATASETS))
    args = parser.parse_args()
    for name in args.datasets.split(","):
        for split in ("train", "test"):
            ds = build_transfer_dataset(name, split, os.path.join(args.root, name))
            print(f"{name:12s} {split:5s} {len(ds):7d} images")
