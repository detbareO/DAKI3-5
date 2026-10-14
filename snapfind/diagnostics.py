import matplotlib

matplotlib.use("Agg")  # kører over SSH uden skærm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import write_json
from .metrics import ap_at_k, mean_or_nan
from .search import l2_normalise


def near_duplicate_similarities(features, query_rows, gallery_rows, model_ids):
    rows_of_model = {}
    for row in gallery_rows:
        rows_of_model.setdefault(model_ids[row], []).append(row)

    similarities = np.empty(len(query_rows))
    for i, query in enumerate(query_rows):
        same_model = l2_normalise(features[rows_of_model[model_ids[query]]])
        similarities[i] = (same_model @ l2_normalise(features[query])).max()
    return similarities


def label_shuffle_check(retrieved, query_rows, gallery_rows, model_ids, k, seed):
    found = retrieved >= 0
    real = (model_ids[retrieved] == model_ids[query_rows][:, None]) & found

    shuffled_ids = model_ids.copy()
    shuffled_ids[gallery_rows] = np.random.default_rng(seed).permutation(model_ids[gallery_rows])
    shuffled = (shuffled_ids[retrieved] == model_ids[query_rows][:, None]) & found

    # tilfældig mAP@k er ca. p * (1 + 1/2 + ... + 1/k)
    images_per_model = pd.Series(model_ids[gallery_rows]).value_counts()
    p = images_per_model.reindex(model_ids[query_rows]).to_numpy() / len(gallery_rows)
    chance = float(p.mean() * sum(1 / i for i in range(1, k + 1)))
    limit = max(0.05, 3 * chance)

    shuffled_map = mean_or_nan(ap_at_k(shuffled))
    return {
        "map_real": mean_or_nan(ap_at_k(real)),
        "map_shuffled": shuffled_map,
        "chance_estimate": chance,
        "limit": limit,
        "passed": bool(shuffled_map <= limit),
    }


def count_duplicate_groups(split, duplicates_path):
    if not duplicates_path.exists():
        return None
    duplicates = pd.read_csv(duplicates_path)[["md5", "path"]]
    merged = duplicates.merge(split[["path", "model_id", "model_group"]], on="path")
    merged = merged[merged["model_group"] != "unused"]

    groups = merged.groupby("md5")
    is_group = groups.size() >= 2
    across_models = groups["model_id"].nunique() > 1
    has_train = groups["model_group"].agg(lambda g: (g == "train").any()).astype(bool)
    has_held_out = groups["model_group"].agg(lambda g: g.isin(["val_unseen", "test_unseen"]).any()).astype(bool)
    return {
        "groups": int(is_group.sum()),
        "groups_across_models": int((is_group & across_models).sum()),
        "groups_across_train_and_held_out": int((is_group & has_train & has_held_out).sum()),
    }


def plot_near_duplicates(similarities, seen, threshold, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.5, 1.0, 51)
    ax.hist(similarities[seen], bins=bins, alpha=0.6, color="tab:blue", label="Seen car models")
    ax.hist(similarities[~seen], bins=bins, alpha=0.6, color="tab:orange", label="Unseen car models")
    ax.axvline(threshold, ls="--", color="black", label=f"Threshold {threshold}")
    ax.set_title("Most similar gallery image of the same car model")
    ax.set_xlabel("Cosine similarity (float A)")
    ax.set_ylabel("Test queries")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "near_duplicates.pdf")
    fig.savefig(out_dir / "near_duplicates.png", dpi=200)
    plt.close(fig)


def run(features, split, query_rows, gallery_rows, retrieved, cfg, out_dir):
    model_ids = split["model_id"].to_numpy()
    seen = split["model_group"].to_numpy()[query_rows] == "train"
    threshold = cfg["eval"]["near_duplicate_threshold"]

    similarities = near_duplicate_similarities(features, query_rows, gallery_rows, model_ids)
    pd.DataFrame({
        "row": query_rows,
        "path": split["path"].to_numpy()[query_rows],
        "group": np.where(seen, "seen", "unseen"),
        "max_similarity_same_model": similarities,
    }).to_csv(out_dir / "near_duplicates.csv", index=False)
    plot_near_duplicates(similarities, seen, threshold, out_dir)

    def share_above(values):
        return float((values > threshold).mean()) if len(values) else None

    report = {
        "near_duplicates": {
            "threshold": threshold,
            "share_above_seen": share_above(similarities[seen]),
            "share_above_unseen": share_above(similarities[~seen]),
            "share_above_all": share_above(similarities),
        },
        "label_shuffle": label_shuffle_check(
            retrieved, query_rows, gallery_rows, model_ids, cfg["search"]["k"], cfg["seed"]),
    }
    write_json(out_dir / "diagnostics.json", report)
    return report
