"""Download a small, reproducible subset of the objects and HDRIs.

The full object dataset is 182 GB and the workstation disk is almost full, so we
never download everything: we read the dataset's index (a table with one row per
object), pick a few objects with a seeded random generator, and fetch only those.

Usage (from the repo root):
    uv run python tools/download_data.py --num-objects 20 --num-hdris 20

Everything goes to ~/data (outside the git repo):
    ~/data/objs/usda/....usda    the objects
    ~/data/objs/index.json       what was downloaded: paths, category, licence, ...
    ~/data/hdris/data/....exr    the environment maps

Running it again with the same arguments downloads nothing new (files already
present are skipped), and gives the same selection: that is what the seed is for.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # reads .parquet files (a compressed table format)
from huggingface_hub import HfApi, hf_hub_download

# Constants: written in UPPER_CASE by convention, never changed at runtime.
OBJ_REPO = "Sandflow/sand3d-objaverse"
HDRI_REPO = "leodriesch/open-hdri-1k"
# The index is split into 13 "shards" of ~120 MB each (~5000 objects per shard).
# One shard is plenty to choose from for now.
OBJ_INDEX_SHARD = "data/objaverse/train-00000-of-00013.parquet"
# Columns we need from the index. The "render" column (a picture of each object)
# is the heavy one, and we skip it.
OBJ_COLUMNS = ["id", "usda_path", "family", "f_score", "n_joints",
               "license", "source_license", "attribution"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "data",
                        help="where to put the data (default: ~/data)")
    parser.add_argument("--num-objects", type=int, default=20)
    parser.add_argument("--num-hdris", type=int, default=20)
    parser.add_argument("--min-f-score", type=float, default=0.85,
                        help="keep objects whose shape matches the original at least this well (0-1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="same seed -> same selection of objects and HDRIs")
    return parser.parse_args()


def download_objects(out_dir: Path, num: int, min_f_score: float, rng: np.random.Generator):
    # 1. Download the index shard (skipped if already there).
    shard_path = hf_hub_download(OBJ_REPO, OBJ_INDEX_SHARD, repo_type="dataset",
                                 local_dir=out_dir)

    # 2. Read it as a list of dicts: [{"id": ..., "usda_path": ..., ...}, ...]
    rows = pq.read_table(shard_path, columns=OBJ_COLUMNS).to_pylist()

    # 3. Keep rigid objects (no joints, so no moving parts) with a good shape.
    #    This is a "list comprehension": a compact for-loop that builds a list.
    good = [r for r in rows if r["n_joints"] == 0 and r["f_score"] >= min_f_score]
    print(f"index: {len(rows)} objects, {len(good)} rigid with f_score >= {min_f_score}")

    # 4. Pick `num` of them at random, without repetitions.
    #    rng.choice returns indices; the selection depends only on the seed.
    picked = rng.choice(len(good), size=min(num, len(good)), replace=False)
    chosen = [good[i] for i in picked]

    # 5. Download each chosen .usda file (skipped if already there).
    for k, row in enumerate(chosen, start=1):
        local = hf_hub_download(OBJ_REPO, row["usda_path"], repo_type="dataset",
                                local_dir=out_dir)
        row["local_path"] = local
        print(f"  [{k}/{len(chosen)}] {row['family']:<22} f={row['f_score']:.2f}  {Path(local).name}")

    # 6. Write down what we have. The licences matter: CC-BY requires attribution
    #    if we ever publish images made from these objects.
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(chosen, indent=2))
    print(f"wrote {index_path}")


def download_hdris(out_dir: Path, num: int, rng: np.random.Generator):
    # List every file in the dataset and keep the 1k EXR images.
    # sorted() matters: the server's order is not guaranteed, and the random
    # choice must run on the same list every time to be reproducible.
    files = HfApi().list_repo_files(HDRI_REPO, repo_type="dataset")
    exrs = sorted(f for f in files if f.endswith("_1k.exr"))
    print(f"hdris: {len(exrs)} available")

    picked = rng.choice(len(exrs), size=min(num, len(exrs)), replace=False)
    for k, i in enumerate(picked, start=1):
        local = hf_hub_download(HDRI_REPO, exrs[i], repo_type="dataset", local_dir=out_dir)
        print(f"  [{k}/{len(picked)}] {Path(local).name}")


def main():
    args = parse_args()
    # Random generators created from the seed: every random choice comes from them.
    # Two separate ones, so that asking for more objects does not change which
    # HDRIs get picked (with a single shared generator it would).
    download_objects(args.data_dir / "objs", args.num_objects, args.min_f_score,
                     np.random.default_rng(args.seed))
    download_hdris(args.data_dir / "hdris", args.num_hdris,
                   np.random.default_rng(args.seed))


if __name__ == "__main__":
    main()
