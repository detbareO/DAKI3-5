import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import read_json, repo_path, sha256_of_file, write_json
from .data import (EVAL_TRANSFORM_DESCRIPTION, CarDataset, degraded_eval_transform, eval_transform,
                   load_index, load_split)
from .models import BACKBONE_ID, FEATURE_DIM, build_backbone, encode_images, load_checkpoint, pick_device


def extract(model, dataset, device, batch_size, workers, out):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = model.to(device).eval()
    done = 0
    with torch.no_grad():
        for batch, (images, rows) in enumerate(loader):
            out[rows.numpy()] = encode_images(model, images.to(device)).cpu().numpy()
            done += len(rows)
            if batch % 20 == 0:
                print(f"  {done:,} / {len(dataset):,} images")


def reusable(features_path, meta_path, n_rows, index_sha, split_sha, complete, model_id):
    if not features_path.exists():
        return False
    if not meta_path.exists():
        # gammel fil uden meta
        if complete and np.load(features_path, mmap_mode="r").shape[0] == n_rows:
            print(f"WARNING: reusing {features_path} without features_meta.json (row count matches)")
            return True
        return False
    meta = read_json(meta_path)
    same_source = (meta.get("index_sha256") == index_sha and meta.get("model") == model_id
                   and meta.get("n_rows") == n_rows)
    return same_source and (bool(meta.get("complete")) or meta.get("split_sha256") == split_sha)


def run(cfg, out_dir, split_path, force=False, checkpoint=None, degraded=False):
    index_path = repo_path(cfg["paths"]["index"])
    index = load_index(index_path)
    index_sha = sha256_of_file(index_path)

    # ved --limit kun de rækker der bruges
    rows = np.arange(len(index))
    split_sha = None
    if split_path.exists():
        split = load_split(split_path)
        if len(split) != len(index) or not (split["path"] == index["path"]).all():
            raise SystemExit(f"{split_path} does not match {index_path}. Run the split stage again.")
        rows = np.flatnonzero((split["role"] != "unused").to_numpy())
        split_sha = sha256_of_file(split_path)
    if checkpoint:
        model = load_checkpoint(checkpoint)
        model_id = f"checkpoint {checkpoint}, sha256 {sha256_of_file(checkpoint)}"
        dim = model.bits or FEATURE_DIM  # C: K værdier af h. A_ft: 512.
    else:
        model, model_id, dim = build_backbone(), BACKBONE_ID, FEATURE_DIM

    transform, name, description = eval_transform(), "features", EVAL_TRANSFORM_DESCRIPTION
    if degraded:
        # Kun test queries, med den faste forringelse. Gallery forbliver rent.
        rows = np.flatnonzero((split["role"] == "test_query").to_numpy())
        transform, name = degraded_eval_transform(), "features_degraded"
        # De forringede queries skal komme fra samme network som de rene features i mappen
        clean_meta = out_dir / "features_meta.json"
        if not clean_meta.exists() or read_json(clean_meta).get("model") != model_id:
            raise SystemExit(f"{out_dir} has no clean features from this network. Run features with "
                             "the same --checkpoint (or none for A) first.")
        description = "fixed degradation (1/3 resolution, blur, 30 % darker and flatter), see data.py"
    complete = len(rows) == len(index)

    features_path = out_dir / f"{name}.npy"
    meta_path = out_dir / f"{name}_meta.json"
    if not force and not degraded:
        if reusable(features_path, meta_path, len(index), index_sha, split_sha, complete, model_id):
            print(f"Reusing {features_path}")
            return
        if features_path.exists():
            raise SystemExit(f"{features_path} exists but does not fit this index or split. "
                             "Pass --force to overwrite it.")

    device = pick_device()
    print(f"Extracting features for {len(rows):,} images on {device}")
    dataset = CarDataset(index, rows, repo_path(cfg["paths"]["data_root"]), transform)


    # skriv til en midlertidig fil først, hvis kørslen bliver afbrudt
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{name}.partial.npy"
    out = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                    shape=(len(index), dim))
    extract(model, dataset, device, cfg["features"]["batch_size"],
            cfg["features"]["workers"], out)
    out.flush()
    del out
    os.replace(partial_path, features_path)

    write_json(meta_path, {
        "model": model_id,
        "index_sha256": index_sha,
        "split_sha256": split_sha,
        "transform": description,
        "n_rows": len(index),
        "n_extracted": len(rows),
        "complete": complete,
        "dim": dim,
        "device": str(device),
    })
    print(f"Saved {features_path}")
