import random

import matplotlib

matplotlib.use("Agg")  # uden skærm: træningen kører over SSH på AAU's GPU
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from . import metrics
from .config import repo_path
from .data import CarDataset, eval_transform, load_index, load_split, train_transform
from .models import RetrievalNet, pick_device
from .split import gallery_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(net, index, rows, data_root, device, batch_size, workers):
    dataset = CarDataset(index, rows, data_root, eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    net.eval()  # batch norm bruger sine gemte statistikker
    parts = []
    with torch.no_grad():
        for images, _ in loader:
            parts.append(net.embed(images.to(device)).cpu().numpy())
    return np.concatenate(parts)


def validate(net, index, split, data_root, device, cfg):
    k = cfg["search"]["k"]
    roles = split["role"].to_numpy()
    query_rows = np.flatnonzero(roles == "val_query")
    gallery_rows = np.flatnonzero(gallery_mask(split, "val"))
    # Samme blanding som i evaluate, så uafgjorte Hamming-afstande brydes tilfældigt
    gallery_rows = gallery_rows[np.random.default_rng(cfg["seed"]).permutation(len(gallery_rows))]

    batch_size, workers = cfg["train"]["batch_size"], cfg["train"]["workers"]
    queries = encode(net, index, query_rows, data_root, device, batch_size, workers)
    gallery = encode(net, index, gallery_rows, data_root, device, batch_size, workers)

    bit_share = None
    if net.bits is None:
        # A_ft: cosine similarity, altså indre produkt mellem vektorer med længde 1
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    else:
        # C: fortegnet giver koden. Som -1 og +1 rangerer indre produkt som Hamming-afstand.
        queries = np.where(queries >= 0, 1.0, -1.0).astype(np.float32)
        gallery = np.where(gallery >= 0, 1.0, -1.0).astype(np.float32)
        bit_share = (gallery > 0).mean(axis=0)  # andel 1-taller for hver bit

    top = metrics.top_k_by_inner_product(queries, gallery, k)
    retrieved = gallery_rows[top]  # pladser i gallery -> rækker i index.csv

    seen = split["model_group"].to_numpy()[query_rows] == "train"
    scores = {}
    for level, column in (("model", "model_id"), ("make", "make_id")):
        labels = split[column].to_numpy()
        rel = labels[retrieved] == labels[query_rows][:, None]
        ap = metrics.ap_at_k(rel)
        scores[f"val_seen_map_{level}"] = metrics.mean_or_nan(ap[seen])
        scores[f"val_unseen_map_{level}"] = metrics.mean_or_nan(ap[~seen])
    return scores, bit_share


def plot_curves(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["epoch"], history["loss_ce"], marker="o", label="cross-entropy")
    if history["loss_quant"].notna().any():
        ax1.plot(history["epoch"], history["loss_quant"], marker="o", label="quantisation (before lambda)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for column in ("val_seen_map_model", "val_unseen_map_model"):
        ax2.plot(history["epoch"], history[column], marker="o", label=column)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation mAP@10")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(cfg, split_path, variant, bits, run_name, epochs=None, max_steps=None):
    train_cfg = cfg["train"]
    set_seed(cfg["seed"])
    if variant == "Aft":
        bits = None
    run_name = run_name or ("Aft" if variant == "Aft" else f"C_{bits}")
    run_dir = repo_path(cfg["paths"]["runs"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs or train_cfg["epochs"]

    index = load_index(repo_path(cfg["paths"]["index"]))
    split = load_split(split_path)
    data_root = repo_path(cfg["paths"]["data_root"])

    # Klasser: hver train car model får et nummer fra 0 til N-1
    train_rows = np.flatnonzero(split["role"].to_numpy() == "train")
    class_ids = np.unique(split["model_id"].to_numpy()[train_rows].astype(int))  # sorteret, hver én gang
    class_of_model = {model_id: number for number, model_id in enumerate(class_ids)}
    label_of_row = np.full(len(split), -1)
    for row in train_rows:
        label_of_row[row] = class_of_model[int(split["model_id"].iloc[row])]
    n_classes = len(class_ids)

    device = pick_device()
    net = RetrievalNet(n_classes, bits).to(device)
    print(f"Training {run_name} on {device}: {len(train_rows):,} images of {n_classes} car models, "
          f"{epochs} epochs")

    dataset = CarDataset(index, train_rows, data_root, train_transform())
    # drop_last: et sidste batch med ét billede ville give batch norm problemer
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                        num_workers=train_cfg["workers"], drop_last=True)

    # Backbonen er fortrænet og får en lille learning rate. De nye lag får 10 gange mere.
    head = [p for name, p in net.named_parameters() if not name.startswith("backbone.")]
    lr = train_cfg["lr_backbone"]
    optimizer = torch.optim.SGD([
        {"params": net.backbone.parameters(), "lr": lr},
        {"params": head, "lr": lr * train_cfg["lr_head_multiplier"]},
    ], momentum=train_cfg["momentum"], weight_decay=train_cfg["weight_decay"])
    # Learning rate falder som en cosinuskurve hen over epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    cross_entropy = torch.nn.CrossEntropyLoss()
    quant_lambda = train_cfg["quant_lambda"]

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({**cfg, "run": {"variant": variant, "bits": bits, "split": str(split_path),
                                       "epochs": epochs, "max_steps": max_steps}}, f)

    history = []
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        net.train()
        total_ce, total_quant, correct, seen_images, steps = 0.0, 0.0, 0, 0, 0
        for images, rows in loader:
            images = images.to(device)
            labels = torch.as_tensor(label_of_row[rows.numpy()], device=device)

            embedding, logits = net(images)
            loss_ce = cross_entropy(logits, labels)
            loss = loss_ce
            if bits is not None:
                # Straf for værdier langt fra -1 og +1
                loss_quant = (embedding.abs() - 1).abs().mean()
                loss = loss_ce + quant_lambda * loss_quant
                total_quant += loss_quant.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_ce += loss_ce.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen_images += len(labels)
            steps += 1
            if steps % 50 == 0:
                print(f"  epoch {epoch}, step {steps}: cross-entropy {total_ce / steps:.3f}")
            if max_steps and steps >= max_steps:
                break
        scheduler.step()

        scores, bit_share = validate(net, index, split, data_root, device, cfg)
        row = {
            "epoch": epoch,
            "loss_ce": total_ce / steps,
            "loss_quant": total_quant / steps if bits is not None else float("nan"),
            "train_accuracy": correct / seen_images,
            **scores,
            "lr_backbone": optimizer.param_groups[0]["lr"],
        }
        if bit_share is not None:
            # En bit, der næsten altid er 0 eller altid 1, bærer ingen information
            row["dead_bits"] = int(((bit_share < 0.05) | (bit_share > 0.95)).sum())
        history.append(row)
        print(f"Epoch {epoch}: " + ", ".join(f"{key} {value:.3f}" for key, value in row.items()
                                              if key != "epoch"))

        checkpoint = {"weights": net.state_dict(), "n_classes": n_classes, "bits": bits,
                      "variant": variant, "epoch": epoch, "class_ids": class_ids.tolist()}
        torch.save(checkpoint, run_dir / "last.pt")
        score = row[train_cfg["select_by"]]
        if score > best_score:
            best_score = score
            torch.save(checkpoint, run_dir / "best.pt")
            print(f"  new best {train_cfg['select_by']}: {score:.3f}, saved best.pt")

        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
        plot_curves(pd.DataFrame(history), run_dir / "curves.png")

    print(f"Done. Best {train_cfg['select_by']}: {best_score:.3f}. Files in {run_dir}")
