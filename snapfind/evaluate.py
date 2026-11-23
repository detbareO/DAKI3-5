import matplotlib

matplotlib.use("Agg")  # kører over SSH uden skærm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import diagnostics, hashing, metrics, search
from .config import repo_path
from .data import VIEWPOINT_NAMES, load_split
from .split import gallery_mask

LEVELS = ["model", "make"]
LEVEL_TITLES = {"model": "Car model", "make": "Make"}
GROUPS = ["seen", "unseen", "all"]
VARIANTS = ["A", "B", "Aft", "Bft", "C"]
# float mod 64 bit
PAIRS = [
    ("A_flat", "B_lsh_64"), ("A_flat", "B_pca_64"),        # Phase 1: float mod naiv hash
    ("Aft_flat", "A_flat"),                                 # hjælper fine-tuning den flydende vektor?
    ("C_64", "B_pca_64"),                                   # R7: slår C den naive PCA-hash?
    ("C_64", "Bft_pca_64"),                                 # er det træningen med hash-lag, der hjælper?
    ("Aft_flat", "C_64"),                                   # hvad koster koden i forhold til float?
]
# Kurverne i map_vs_bits: (navnets start, forklaring, farve)
CURVES = [
    ("B_lsh_", "LSH hash of A", "tab:blue"),
    ("B_pca_", "PCA hash of A", "tab:orange"),
    ("Bft_lsh_", "LSH hash of A_ft", "tab:purple"),
    ("Bft_pca_", "PCA hash of A_ft", "tab:red"),
    ("C_", "C (deep hash)", "tab:green"),
]


def search_one(variant, representation, index, index_name, bits, queries, gallery_rows, cfg):
    k, repeats = cfg["search"]["k"], cfg["search"]["timing_repeats"]
    ids, ms_per_query = search.timed_search(index, queries, k, repeats)
    ms_single = search.timed_single_queries(index, queries, k, repeats)
    result = {
        "variant": variant,
        "backbone": "resnet18",
        "representation": representation,
        "index": index_name,
        "bits": bits,
        "gallery_MB": search.index_bytes(index) / 1e6,
        "search_ms_per_query": ms_per_query,
        "search_ms_single_query": ms_single,
        # FAISS-id'er er pladser i det blandede gallery: oversæt dem tilbage til rækker i index.csv
        "retrieved": np.where(ids >= 0, gallery_rows[ids], -1),
    }
    print(f"  {variant}: {ms_per_query:.3f} ms per query in one batch, {ms_single:.3f} ms one at a time, "
          f"{result['gallery_MB']:.2f} MB")
    return result


def search_variants(features, roles, query_rows, gallery_rows, cfg, float_name, hash_name, with_hash):
    seed = cfg["seed"]
    dim = features.shape[1]
    found = []

    def add(variant, representation, index, index_name, bits, queries):
        found.append(search_one(variant, representation, index, index_name, bits, queries, gallery_rows, cfg))

    # normaliseret, så indre produkt = cosine
    gallery_features, query_features = features[gallery_rows], features[query_rows]
    gallery = search.l2_normalise(gallery_features)
    queries = search.l2_normalise(query_features)
    add(f"{float_name}_flat", "float", search.build_flat_ip(gallery), "IndexFlatIP", dim * 32, queries)
    if not with_hash:
        return found

    # PCA fittes kun på train
    mean, components = hashing.fit_on_train(features, roles, cfg["hashing"]["pca_fit_max_rows"], seed)
    for bits in cfg["hashing"]["bits"]:
        projections = {"lsh": hashing.lsh_matrix(dim, bits, seed), "pca": components[:, :bits]}
        for method, projection in projections.items():
            gallery_codes = hashing.encode(gallery_features, mean, projection)
            query_codes = hashing.encode(query_features, mean, projection)
            add(f"{hash_name}_{method}_{bits}", method, search.build_binary(gallery_codes, bits),
                "IndexBinaryFlat", bits, query_codes)
    return found


def score(retrieved, query_rows, labels):
    rel = (labels[retrieved] == labels[query_rows][:, None]) & (retrieved >= 0)
    return metrics.ap_at_k(rel), metrics.p_at_k(rel)


def save_figure(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def search_deep_hash(folder, bits, query_rows, gallery_rows, cfg, degraded):
    values = load_features(folder, query_rows, gallery_rows, degraded)
    gallery_codes = hashing.sign_codes(values[gallery_rows])
    query_codes = hashing.sign_codes(values[query_rows])
    return search_one(f"C_{bits}", "deep hash", search.build_binary(gallery_codes, bits),
                      "IndexBinaryFlat", bits, query_codes, gallery_rows, cfg)


def load_features(folder, query_rows, gallery_rows, degraded):
    features = np.load(folder / "features.npy")
    if degraded:
        # Kun queries forringes. Gallery er katalogets rene billeder.
        features[query_rows] = np.load(folder / "features_degraded.npy")[query_rows]
    # Rækker, der aldrig blev udtrukket, er rene nuller (features fra et andet --limit-split)
    needed = np.concatenate([query_rows, gallery_rows])
    if not np.abs(features[needed]).sum(axis=1).all():
        raise SystemExit(f"Some query or gallery rows have no features in {folder}. "
                         "Run the features stage with this split.")
    return features


def plot_map_vs_bits(results, k, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for r, level in enumerate(LEVELS):
        for c, group in enumerate(["seen", "unseen"]):
            ax, column = axes[r, c], f"map{k}_{level}_{group}"
            for prefix, label, colour in CURVES:
                part = results[results["variant"].str.startswith(prefix)].sort_values("bits")
                if part.empty:
                    continue
                errors = np.array([part[column] - part[column + "_lo"], part[column + "_hi"] - part[column]])
                ax.errorbar(part["bits"], part[column], yerr=np.clip(errors, 0, None), marker="o",
                            capsize=3, color=colour, label=label)
                ax.set_xscale("log", base=2)
                ax.set_xticks(part["bits"].tolist(), labels=[str(b) for b in part["bits"]])
                ax.minorticks_off()
                # ellers bliver der tom plads til venstre
                ax.set_xlim(part["bits"].min() / 1.4, part["bits"].max() * 1.4)
            # Vandrette linjer: float-vektorerne, som koderne sammenlignes med
            for variant, style, label in (("A_flat", "--", "Float A (512-d, exact)"),
                                          ("Aft_flat", ":", "Float A_ft (512-d, exact)")):
                reference = results[results["variant"] == variant]
                if not reference.empty:
                    ax.axhline(reference[column].iloc[0], ls=style, color="black", label=label)
            ax.set_title(f"{LEVEL_TITLES[level]} level, {group} car models")
            ax.set_ylabel(f"mAP@{k}")
            ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Code length (bits)")
    axes[0, 0].legend(loc="lower right")
    save_figure(fig, out_dir, "map_vs_bits")


def plot_search_cost(results, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Én query ad gangen, som når én telefon sender én query
    ax1.bar(results["variant"], results["search_ms_single_query"], color="tab:blue")
    ax1.set_title("Search time, one query at a time (1 thread)")
    ax1.set_ylabel("ms (log scale)")
    ax2.bar(results["variant"], results["gallery_MB"], color="tab:orange")
    ax2.set_title("Gallery index size")
    ax2.set_ylabel("MB (log scale)")
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    save_figure(fig, out_dir, "search_cost")


def print_checks(results, k):
    column = f"map{k}_model"
    apart = results[(results[f"{column}_seen_lo"] > results[f"{column}_unseen_hi"])
                    | (results[f"{column}_unseen_lo"] > results[f"{column}_seen_hi"])]
    print("Seen and unseen intervals overlap for every variant (expected for A and B, which are not trained)"
          if apart.empty else
          f"Seen and unseen intervals do NOT overlap for: {apart['variant'].tolist()}. "
          "Expected for trained variants. For A and B, inspect the split.")

    float_map = results.loc[results["variant"] == "A_flat", f"{column}_all"].iloc[0]
    for method in ("lsh", "pca"):
        part = results[results["variant"].str.startswith(f"B_{method}_")].sort_values("bits")
        if part.empty:
            continue
        values = part[f"{column}_all"].to_numpy()
        print(f"B_{method}: at or below A for every K: {bool((values <= float_map).all())}, "
              f"improves as K grows: {bool((np.diff(values) > 0).all())}")


def run(cfg, out_dir, split_path, variants=("A", "B"), feature_dirs=None, degraded=False):
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown variants {sorted(unknown)}. Choose from {VARIANTS}.")
    feature_dirs = feature_dirs or {}

    def folder_of(name, default):
        return repo_path(feature_dirs.get(name, default))

    k, seed = cfg["search"]["k"], cfg["seed"]
    n_resamples = cfg["eval"]["bootstrap_resamples"]
    split = load_split(split_path)

    roles = split["role"].to_numpy()
    model_ids, make_ids = split["model_id"].to_numpy(), split["make_id"].to_numpy()
    query_rows = np.flatnonzero(roles == "test_query")
    # blandes så FAISS ikke bryder uafgjorte efter rækkefølgen i index.csv
    gallery_rows = np.flatnonzero(gallery_mask(split, "test"))
    gallery_rows = gallery_rows[search.shuffle_order(len(gallery_rows), seed)]

    features = load_features(folder_of("A", cfg["feature_dirs"]["A"]), query_rows, gallery_rows, degraded)
    if len(features) != len(split):
        raise SystemExit("features.npy and split.csv have different row counts")

    seen = split["model_group"].to_numpy()[query_rows] == "train"
    group_masks = {"seen": seen, "unseen": ~seen, "all": np.ones(len(query_rows), dtype=bool)}
    viewpoints = split["viewpoint"].to_numpy()[query_rows]
    print(f"{seen.sum()} seen and {(~seen).sum()} unseen queries, gallery of {len(gallery_rows):,} images")

    found = search_variants(features, roles, query_rows, gallery_rows, cfg, "A", "B", "B" in variants)
    if "Aft" in variants or "Bft" in variants:
        features_ft = load_features(folder_of("Aft", cfg["feature_dirs"]["Aft"]), query_rows, gallery_rows, degraded)
        found += search_variants(features_ft, roles, query_rows, gallery_rows, cfg, "Aft", "Bft", "Bft" in variants)
    if "C" in variants:
        for bits in cfg["hashing"]["bits"]:
            folder = folder_of(f"C_{bits}", cfg["feature_dirs"]["C"].format(bits=bits))
            if not (folder / "features.npy").exists():
                print(f"  C_{bits}: no features in {folder}, skipped")
                continue
            found.append(search_deep_hash(folder, bits, query_rows, gallery_rows, cfg, degraded))

    result_rows, viewpoint_rows = [], []
    per_query = {}
    for variant in found:
        name = variant["variant"]
        scores, bounds = {}, {}
        for level, labels in (("model", model_ids), ("make", make_ids)):
            ap, p = score(variant["retrieved"], query_rows, labels)
            per_query[f"{name}__{level}"] = ap
            for group, mask in group_masks.items():
                column = f"map{k}_{level}_{group}"
                scores[column] = metrics.mean_or_nan(ap[mask])
                scores[f"p{k}_{level}_{group}"] = metrics.mean_or_nan(p[mask])
                bounds[f"{column}_lo"], bounds[f"{column}_hi"] = metrics.bootstrap_ci(ap[mask], n_resamples, seed)

            if level == "model":
                for group, mask in group_masks.items():
                    for viewpoint in np.unique(viewpoints):
                        chosen = mask & (viewpoints == viewpoint)
                        viewpoint_rows.append({
                            "variant": name, "group": group, "viewpoint": int(viewpoint),
                            "viewpoint_name": VIEWPOINT_NAMES.get(int(viewpoint), "?"),
                            "n_queries": int(chosen.sum()),
                            f"map{k}_model": metrics.mean_or_nan(ap[chosen]),
                            f"p{k}_model": metrics.mean_or_nan(p[chosen]),
                        })

        row = {key: variant[key] for key in ("variant", "backbone", "representation", "index", "bits")}
        row.update(scores)
        row.update(bounds)
        row.update({
            "gallery_MB": variant["gallery_MB"],
            "search_ms_per_query": variant["search_ms_per_query"],
            "search_ms_single_query": variant["search_ms_single_query"],
            "n_query_seen": int(seen.sum()),
            "n_query_unseen": int((~seen).sum()),
            "n_gallery": len(gallery_rows),
        })
        result_rows.append(row)

    # parret bootstrap
    pair_rows = []
    for a, b in PAIRS:
        if f"{a}__model" not in per_query or f"{b}__model" not in per_query:
            continue
        for level in LEVELS:
            ap_a, ap_b = per_query[f"{a}__{level}"], per_query[f"{b}__{level}"]
            for group, mask in group_masks.items():
                difference, low, high = metrics.paired_bootstrap(ap_a[mask], ap_b[mask], n_resamples, seed)
                pair_rows.append({
                    "variant_a": a, "variant_b": b, "level": level, "group": group,
                    f"map{k}_a": metrics.mean_or_nan(ap_a[mask]), f"map{k}_b": metrics.mean_or_nan(ap_b[mask]),
                    "difference": difference, "difference_lo": low, "difference_hi": high,
                    "n_queries": int(mask.sum()),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(viewpoint_rows).to_csv(out_dir / "results_by_viewpoint.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "paired_differences.csv", index=False)
    np.savez(out_dir / "per_query_ap.npz", query_rows=query_rows, seen=seen, **per_query)
    plot_map_vs_bits(results, k, out_dir)
    plot_search_cost(results, out_dir)

    print(results[["variant", "bits", f"map{k}_model_seen", f"map{k}_model_unseen",
                   f"map{k}_make_unseen", "gallery_MB", "search_ms_per_query",
                   "search_ms_single_query"]].to_string(index=False))
    print_checks(results, k)

    report = diagnostics.run(features, split, query_rows, gallery_rows, found[0]["retrieved"], cfg, out_dir)
    print("Near-duplicate share (seen / unseen):", report["near_duplicates"]["share_above_seen"],
          "/", report["near_duplicates"]["share_above_unseen"])
    shuffle = report["label_shuffle"]
    print(f"Label shuffle: mAP {shuffle['map_real']:.3f} -> {shuffle['map_shuffled']:.3f} "
          f"(limit {shuffle['limit']:.3f})")
    if not shuffle["passed"]:
        raise RuntimeError("Label-shuffle check FAILED: mAP did not fall to chance level with permuted "
                           "labels. Rows are probably misaligned (features vs. split).")
    print(f"Saved results to {out_dir}")
