from datetime import datetime
import io
import time

import numpy as np
import pandas as pd
import torch

from .config import repo_path, sha256_of_file
from .data import eval_resize, eval_transform, load_crop, load_index, load_split
from .models import BACKBONE_ID, FEATURE_DIM, build_backbone, encode_images, load_checkpoint, pick_device

WARM_UP_RUNS = 10


def time_encoding(model, tensors, device):
    model = model.to(device).eval()
    times = []
    with torch.no_grad():
        # opvarmning, tælles ikke med
        for _ in range(WARM_UP_RUNS):
            encode_images(model, tensors[0].to(device)).cpu()
        for tensor in tensors:
            start = time.perf_counter()
            # .cpu() venter på at GPU'en er færdig
            encode_images(model, tensor.to(device)).cpu()
            times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times)), float(np.percentile(times, 95))


def run(cfg, out_dir, split_path, device_label, checkpoint=None):
    index = load_index(repo_path(cfg["paths"]["index"]))
    rows = np.arange(len(index))
    if split_path.exists():
        rows = np.flatnonzero((load_split(split_path)["role"] != "unused").to_numpy())
    rng = np.random.default_rng(cfg["seed"])
    rows = np.sort(rng.choice(rows, size=min(cfg["costs"]["n_timing"], len(rows)), replace=False))

    data_root = repo_path(cfg["paths"]["data_root"])
    paths = index["path"].to_numpy()
    boxes = index[["x1", "y1", "x2", "y2"]].to_numpy()
    transform, resize = eval_transform(), eval_resize()

    tensors, preprocess_ms, jpeg_bytes = [], [], []
    for row in rows:
        start = time.perf_counter()
        crop = load_crop(data_root, paths[row], boxes[row])
        tensors.append(transform(crop).unsqueeze(0))
        preprocess_ms.append(1000 * (time.perf_counter() - start))

        # størrelsen på billedet hvis det blev sendt som JPEG
        buffer = io.BytesIO()
        resize(crop).save(buffer, format="JPEG", quality=cfg["costs"]["jpeg_quality"])
        jpeg_bytes.append(len(buffer.getvalue()))

    if checkpoint:
        # Et trænet network fra train.py, fx C: ResNet-18 plus ét lille lineært lag
        model = load_checkpoint(checkpoint)
        model_id = f"checkpoint {checkpoint}, sha256 {sha256_of_file(checkpoint)}"
    else:
        model, model_id = build_backbone(), BACKBONE_ID
    threads = cfg["costs"]["cpu_threads"]
    torch.set_num_threads(threads)
    devices = [(torch.device("cpu"), threads)]
    if pick_device().type != "cpu":
        devices.append((pick_device(), ""))

    result_rows = []
    for device, used_threads in devices:
        median, p95 = time_encoding(model, tensors, device)
        result_rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device_label": device_label,
            "device": device.type,
            "cpu_threads": used_threads,
            "model": model_id,
            "n_images": len(rows),
            "encode_ms_median": median,
            "encode_ms_p95": p95,
            "preprocess_ms_median": float(np.median(preprocess_ms)),
            "jpeg_bytes_median": float(np.median(jpeg_bytes)),
            "float_vector_bytes": FEATURE_DIM * 4,
        })
        print(f"  {device_label} / {device.type}: median {median:.1f} ms, p95 {p95:.1f} ms per image")

    # append så alle maskiner ender i samme fil
    out_dir.mkdir(parents=True, exist_ok=True)
    costs_path = out_dir / "costs.csv"
    pd.DataFrame(result_rows).to_csv(costs_path, mode="a", header=not costs_path.exists(), index=False)
    print(f"Median JPEG size of the crop: {np.median(jpeg_bytes):.0f} bytes. Appended to {costs_path}")
