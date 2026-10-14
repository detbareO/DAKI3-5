import numpy as np
import pandas as pd

from .config import repo_path, sha256_of_file, write_json
from .data import load_index
from .diagnostics import count_duplicate_groups

HELD_OUT = ["val_unseen", "test_unseen"]
QUERY_ROLES = ["test_query", "val_query"]
GALLERY_ROLES = ["train", "gallery_only"]

# max andel queries, betyder kun noget ved --limit
MAX_QUERY_SHARE = 0.25
SUBSET_MODELS_PER_MAKE = 5


def pick_subset_models(index, limit, rng):
    images_per_model = index.groupby("model_id").size()
    models_of_make = index.groupby("make_id")["model_id"].unique()
    chosen, n_images = [], 0
    for make in rng.permutation(np.sort(models_of_make.index.to_numpy())):
        models = rng.permutation(np.sort(models_of_make[make]))[:SUBSET_MODELS_PER_MAKE]
        chosen.extend(models.tolist())
        n_images += int(images_per_model[models].sum())
        if n_images >= limit:
            break
    return np.array(chosen)


def assign_model_groups(models, split_cfg, rng):
    # én car model pr. make går altid til train
    anchors = [int(rng.choice(group["model_id"].to_numpy())) for _, group in models.groupby("make_id")]

    # resten blandes: 20 % test_unseen, 10 % val_unseen
    rest = rng.permutation(np.setdiff1d(models["model_id"].to_numpy(), anchors))
    n_test = round(split_cfg["test_unseen_share"] * len(models))
    n_val = round(split_cfg["val_unseen_share"] * len(models))
    if n_test + n_val > len(rest):
        scale = len(rest) / (n_test + n_val)
        n_test, n_val = int(n_test * scale), int(n_val * scale)
        print(f"Too few non-anchor car models: holding out {n_test} test and {n_val} validation car models")

    group_of_model = dict.fromkeys(models["model_id"].tolist(), "train")
    group_of_model.update(dict.fromkeys(rest[:n_test].tolist(), "test_unseen"))
    group_of_model.update(dict.fromkeys(rest[n_test:n_test + n_val].tolist(), "val_unseen"))
    return group_of_model


def scale_counts(pool_size, wanted):
    cap = int(pool_size * MAX_QUERY_SHARE)
    if sum(wanted) <= cap:
        return list(wanted)
    return [int(n * cap / sum(wanted)) for n in wanted]


def make_split(index, split_cfg, seed, limit=None):
    rng = np.random.default_rng(seed)
    model_ids = index["model_id"].to_numpy()

    used = np.ones(len(index), dtype=bool)
    if limit is not None and limit < len(index):
        used = np.isin(model_ids, pick_subset_models(index, limit, rng))

    models = index.loc[used, ["make_id", "model_id"]].drop_duplicates().sort_values(["make_id", "model_id"])
    if not models["model_id"].is_unique:
        raise ValueError("A model_id appears under more than one make")
    group_of_model = assign_model_groups(models, split_cfg, rng)
    model_group = index["model_id"].map(group_of_model).fillna("unused").to_numpy(dtype=object)

    role = np.full(len(index), "unused", dtype=object)
    role[model_group == "train"] = "train"
    role[np.isin(model_group, HELD_OUT)] = "gallery_only"

    # en query skal have mindst ét andet billede af samme car model
    images_per_model = index.loc[used, "model_id"].value_counts()
    can_query = index["model_id"].map(images_per_model).fillna(0).to_numpy() >= split_cfg["min_images_for_query"]

    def draw(group, wanted):
        pool = np.flatnonzero((model_group == group) & can_query)
        counts = scale_counts(len(pool), wanted)
        picked = rng.choice(pool, size=sum(counts), replace=False)
        return np.split(picked, np.cumsum(counts)[:-1])

    n_test, n_val = split_cfg["n_test_queries"], split_cfg["n_val_queries"]
    # samme træk, så de ikke overlapper
    test_seen, val_seen = draw("train", [n_test["seen"], n_val["seen"]])
    (test_unseen,) = draw("test_unseen", [n_test["unseen"]])
    (val_unseen,) = draw("val_unseen", [n_val["unseen"]])
    role[test_seen] = "test_query"
    role[test_unseen] = "test_query"
    role[val_seen] = "val_query"
    role[val_unseen] = "val_query"

    # car models hvor alle billeder blev queries: flyt én tilbage
    is_query = np.isin(role, QUERY_ROLES)
    left_in_gallery = pd.Series(model_ids[used & ~is_query]).value_counts()
    for model in np.unique(model_ids[is_query]):
        if left_in_gallery.get(model, 0) == 0:
            row = np.flatnonzero((model_ids == model) & is_query).max()
            role[row] = "train" if model_group[row] == "train" else "gallery_only"

    split = index[["path", "make_id", "model_id", "viewpoint"]].copy()
    split["model_group"] = model_group
    split["role"] = role
    return split


def gallery_mask(split, which):
    in_gallery = split["role"].isin(GALLERY_ROLES)
    if which == "val":
        # validation må ikke se test car models
        in_gallery &= split["model_group"].isin(["train", "val_unseen"])
    return in_gallery.to_numpy()


def check_split(split):
    used = split[split["model_group"] != "unused"]
    group, role = used["model_group"], used["role"]
    held_out = group.isin(HELD_OUT)

    assert group.isin(["train"] + HELD_OUT).all(), "unknown car model group"
    assert (used.groupby("model_id")["model_group"].nunique() == 1).all(), "a car model is in two groups"
    assert not (held_out & (role == "train")).any(), "a held-out car model has train rows"

    assert role[group == "train"].isin(["train"] + QUERY_ROLES).all(), "bad role in a train car model"
    assert role[held_out].isin(["gallery_only"] + QUERY_ROLES).all(), "bad role in a held-out car model"
    assert group[role == "test_query"].isin(["train", "test_unseen"]).all(), "test query in a validation car model"
    assert group[role == "val_query"].isin(["train", "val_unseen"]).all(), "validation query in a test car model"

    train_makes = set(used.loc[group == "train", "make_id"])
    assert set(used.loc[held_out, "make_id"]) <= train_makes, "a held-out car model's make has no train car model"

    for query_role, which in (("test_query", "test"), ("val_query", "val")):
        gallery_models = set(split.loc[gallery_mask(split, which), "model_id"])
        query_models = set(split.loc[split["role"] == query_role, "model_id"])
        assert query_models <= gallery_models, f"a {query_role} has no relevant image in its gallery"


def split_counts(split):
    used = split[split["model_group"] != "unused"]
    seen = used["model_group"] == "train"

    def as_ints(series):
        return {str(key): int(value) for key, value in series.items()}

    return {
        "models_per_group": as_ints(used.groupby("model_group")["model_id"].nunique()),
        "images_per_group": as_ints(used["model_group"].value_counts()),
        "images_per_role": as_ints(used["role"].value_counts()),
        "queries": {
            "test_seen": int(((used["role"] == "test_query") & seen).sum()),
            "test_unseen": int(((used["role"] == "test_query") & ~seen).sum()),
            "val_seen": int(((used["role"] == "val_query") & seen).sum()),
            "val_unseen": int(((used["role"] == "val_query") & ~seen).sum()),
        },
        "unused_images": int((split["model_group"] == "unused").sum()),
    }


def run(cfg, out_dir, limit=None, force=False):
    index_path = repo_path(cfg["paths"]["index"])
    index = load_index(index_path)
    split = make_split(index, cfg["split"], cfg["seed"], limit)
    check_split(split)

    csv_text = split.to_csv(index=False, lineterminator="\n")
    split_path = out_dir / "split.csv"
    if split_path.exists() and split_path.read_text() != csv_text and not force:
        raise SystemExit(f"{split_path} exists and differs from the new split. Pass --force to overwrite it.")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(csv_text)

    counts = split_counts(split)
    write_json(out_dir / "split_meta.json", {
        "version": "v2",
        "seed": cfg["seed"],
        "limit": limit,
        "index_sha256": sha256_of_file(index_path),
        "counts": counts,
        "exact_duplicates": count_duplicate_groups(split, repo_path(cfg["paths"]["duplicates"])),
    })

    print(f"Saved {split_path}")
    for name, values in counts.items():
        print(f"  {name}: {values}")
