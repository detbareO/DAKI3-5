from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image

start = time.time()
BASE = Path(__file__).parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "index.csv")
print(df.columns.tolist())

# Navne på mærker og modeller fra misc/ (id 1 ligger på plads 0)
names = loadmat(BASE / "misc" / "make_model_name.mat")
make_names = [n[0][0] if n[0].size else "?" for n in names["make_names"]]
model_names = [n[0][0] if n[0].size else "?" for n in names["model_names"]]
df["make_name"] = df["make_id"].map(lambda i: make_names[i - 1])
df["model_name"] = df["model_id"].map(lambda i: model_names[i - 1])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


# 1. Billeder pr. model
print("\n== 1. Images per model ==")
df["model"] = df["make_id"].astype(str) + "_" + df["model_id"].astype(str)
counts = df["model"].value_counts()
print("Modeller:", len(counts))
print("Median billeder pr. model:", counts.median())
print("Min / max:", counts.min(), counts.max())
print("Andel billeder i top 100:", round(counts.head(100).sum() / len(df) * 100, 1), "%")
print("Modeller med præcis 120 billeder:", (counts == 120).sum())

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(range(len(counts)), counts.values, color="tab:blue", alpha=0.2, step="post")
ax.plot(range(len(counts)), counts.values, color="tab:blue", lw=1.5)

# Referenceniveauer: kun information, ikke en grænse vi har valgt
for level, color in [(20, "tab:red"), (50, "tab:orange"), (100, "tab:green")]:
    over = counts >= level
    print(f">= {level} billeder: {over.sum()} modeller, {counts[over].sum() / len(df):.1%} af billederne")
    ax.axhline(level, ls="--", color=color, lw=1,
               label=f">= {level} images: {over.sum():,} car models, {counts[over].sum() / len(df):.1%} of images")
ax.axhline(counts.median(), ls=":", color="gray", label=f"Median: {counts.median():.0f} images")

ax.annotate(f"{(counts == 120).sum()} car models have exactly 120 images\n(likely a collection cap)",
            xy=(400, 120), xytext=(450, 300), arrowprops=dict(arrowstyle="->"))
ax.set_yscale("log")
ax.set_ylim(0.8, 600)
ax.set_xlim(0, len(counts))
ax.set_title(f"Images per car model in CompCars ({len(counts):,} car models, {len(df):,} images)")
ax.set_xlabel("Car model (sorted by number of images)")
ax.set_ylabel("Images (log scale)")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left")
save(fig, "images_per_model.png")

# 2. Synsvinkler
print("\n== 2. Viewpoints ==")
vp_names = {-1: "unknown", 1: "front", 2: "rear", 3: "side", 4: "front-side", 5: "rear-side"}
df["vp_name"] = df["viewpoint"].map(vp_names)
vp = df["vp_name"].value_counts()
print((vp / len(df) * 100).round(1).astype(str) + " %")

fig, ax = plt.subplots(figsize=(7, 4))
vp.plot(kind="bar", ax=ax, color="tab:blue")
# Procent over hver søjle
for i, v in enumerate(vp.values):
    ax.text(i, v, f"{v / len(df):.1%}", ha="center", va="bottom")
ax.set_title("Viewpoint distribution")
ax.set_xlabel("Viewpoint")
ax.set_ylabel("Images")
ax.tick_params(axis="x", rotation=0)
save(fig, "viewpoints.png")

# 3. Årstal
print("\n== 3. Years ==")
years = df["year"].astype(str).value_counts().sort_index()
odd = years[~years.index.str.fullmatch(r"20(0\d|1\d)")]  # alt uden for 2000-2019
print("Usædvanlige årstal:\n", odd.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["tab:red" if y in odd.index else "tab:blue" for y in years.index]
ax.bar(years.index, years.values, color=colors)
ax.set_yscale("log")  # ellers er de små søjler usynlige
ax.set_title("Images per model year (red = unusual values)")
ax.set_xlabel("Year")
ax.set_ylabel("Images (log scale)")
ax.tick_params(axis="x", rotation=90)
save(fig, "years.png")

# 4. Billedstørrelse og filstørrelse
print("\n== 4. Image and file size ==")
print("Bredde min / median / max:", df["width"].min(), df["width"].median(), df["width"].max())
print("Højde  min / median / max:", df["height"].min(), df["height"].median(), df["height"].max())
print("Filstørrelse median:", round(df["file_bytes"].median() / 1000, 1), "KB")
print("Samlet størrelse:", round(df["file_bytes"].sum() / 1e9, 2), "GB")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.scatter(df["width"], df["height"], s=1, alpha=0.1)
ax1.set_title("Image resolution")
ax1.set_xlabel("Width (px)")
ax1.set_ylabel("Height (px)")
ax2.hist(df["file_bytes"] / 1000, bins=80, color="tab:blue")
ax2.axvline(df["file_bytes"].median() / 1000, ls="--", color="black",
            label=f"Median: {df['file_bytes'].median() / 1000:.0f} KB")
ax2.set_title("File size")
ax2.set_xlabel("File size (KB)")
ax2.set_ylabel("Images")
ax2.legend()
save(fig, "image_size.png")

# 5. Bounding box: hvor meget af billedet fylder bilen
print("\n== 5. Bounding box ==")
box_area = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
df["bbox_share"] = box_area / (df["width"] * df["height"])
zero = df[box_area <= 0]
print("Bbox-andel min / median / max:", round(df["bbox_share"].min(), 3),
      round(df["bbox_share"].median(), 3), round(df["bbox_share"].max(), 3))
print("Bokse med areal 0:", len(zero))
print(zero[["path", "x1", "y1", "x2", "y2"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["bbox_share"] * 100, bins=50, color="tab:blue")
ax.axvline(df["bbox_share"].median() * 100, ls="--", color="black",
           label=f"Median: {df['bbox_share'].median():.0%}")
ax.set_title("Share of the image covered by the car (bounding box)")
ax.set_xlabel("Bounding box area / image area (%)")
ax.set_ylabel("Images")
ax.legend()
save(fig, "bbox_share.png")

# 6. Mærker
print("\n== 6. Makes ==")
# Tæl pr. make_id, fordi to mærker (133 og 141) begge hedder "Lotus"
makes = df["make_id"].value_counts()
makes.index = [f"{make_names[i - 1]} ({i})" if make_names[i - 1] == "Lotus" else make_names[i - 1]
               for i in makes.index]
models_per_make = df.groupby("make_id")["model"].nunique()
print("Top 10 mærker (billeder):\n", makes.head(10).to_string())
print("Mærker med under 50 billeder:", (makes < 50).sum())
print("Median modeller pr. mærke:", models_per_make.median())

fig, ax = plt.subplots(figsize=(10, 5))
makes.head(30).plot(kind="bar", ax=ax, color="tab:blue")
ax.set_title(f"Top 30 of {len(makes)} makes by number of images")
ax.set_xlabel("Make")
ax.set_ylabel("Images")
save(fig, "makes_top30.png")

# 7. Synsvinkler pr. model
print("\n== 7. Viewpoints per model ==")
known = df[df["viewpoint"] != -1]
vp_table = pd.crosstab(known["model"], known["vp_name"])  # modeller x synsvinkler
n_views = (vp_table > 0).sum(axis=1)
print("Modeller med alle 5 synsvinkler:", (n_views == 5).sum(), "af", len(vp_table))
missing = (vp_table == 0).sum()
print("Modeller der mangler hver synsvinkel:\n", missing.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
n_views.value_counts().sort_index().plot(kind="bar", ax=ax1, color="tab:blue")
ax1.set_title("Number of viewpoints covered per car model")
ax1.set_xlabel("Viewpoints with at least one image")
ax1.set_ylabel("Car models")
ax1.tick_params(axis="x", rotation=0)
missing.sort_values().plot(kind="bar", ax=ax2, color="tab:red")
ax2.set_title("Car models with zero images of a viewpoint")
ax2.set_xlabel("Viewpoint")
ax2.set_ylabel("Car models")
ax2.tick_params(axis="x", rotation=0)
save(fig, "viewpoints_per_model.png")

# 8. Den officielle split
print("\n== 8. Official classification split ==")
split_dir = BASE / "train_test_split" / "classification"
train = set((split_dir / "train.txt").read_text().split())
test = set((split_dir / "test.txt").read_text().split())
df["split"] = df["path"].map(lambda p: "train" if p in train else "test" if p in test else "not used")
print(df["split"].value_counts().to_string())
in_split = df[df["split"] != "not used"]
print("Modeller i split:", in_split["model"].nunique())
per_model_train = in_split[in_split["split"] == "train"]["model"].value_counts()
print("Train-billeder pr. model min / median / max:",
      per_model_train.min(), per_model_train.median(), per_model_train.max())
print("Synsvinkler i split:\n", in_split["vp_name"].value_counts().to_string())

# 9. Bil-typer fra misc/attributes.txt
print("\n== 9. Car types (misc) ==")
attr = pd.read_csv(BASE / "misc" / "attributes.txt", sep=" ")
type_names = ["unknown"] + [t[0] for t in loadmat(BASE / "misc" / "car_type.mat")["types"][0]]
attr["type_name"] = attr["type"].map(lambda t: type_names[t])
df = df.merge(attr[["model_id", "type_name"]], on="model_id", how="left")
print("Modeller pr. type:\n", attr["type_name"].value_counts().to_string())
print("Billeder pr. type:\n", df["type_name"].value_counts().to_string())

fig, ax = plt.subplots(figsize=(9, 4))
df["type_name"].value_counts().plot(kind="bar", ax=ax, color="tab:blue")
ax.set_title("Images per car type")
ax.set_xlabel("Car type")
ax.set_ylabel("Images")
save(fig, "car_types.png")

# 10. Bildele i part/
print("\n== 10. Car parts (part/) ==")
part_names = {1: "headlight", 2: "taillight", 3: "fog light", 4: "air intake",
              5: "console", 6: "steering wheel", 7: "dashboard", 8: "gear lever"}
# Sti: part/<make>/<model>/<year>/<part_type>/<name>.jpg
parts = pd.Series([int(p.parent.name) for p in (BASE / "part").rglob("*.jpg")])
part_counts = parts.map(part_names).value_counts()
print("Billeder af bildele:", len(parts))
print(part_counts.to_string())

# 11. Modeller der næsten er ens (svære par fra den officielle verification-split)
print("\n== 11. Similar-looking models ==")
pairs = pd.read_csv(BASE / "train_test_split" / "verification" / "verification_pairs_hard.txt",
                    sep=" ", header=None, names=["a", "b", "same"])
info = df.set_index("path")[["make_id", "model_name", "vp_name", "type_name"]]
diff = pairs[pairs["same"] == 0].join(info, on="a").join(info, on="b", rsuffix="_b")
print("Svære par med forskellige modeller:", len(diff))
print("Heraf samme mærke:", (diff["make_id"] == diff["make_id_b"]).mean().round(3) * 100, "%")

# Kun par med samme synsvinkel og samme kendte bil-type, så forskellen er selve modellen
alike = diff[(diff["vp_name"] == diff["vp_name_b"]) & (diff["type_name"] == diff["type_name_b"])
             & (diff["type_name"] != "unknown")]
print("Heraf samme synsvinkel og type:", len(alike))
show = alike.sample(6, random_state=42)

fig, axes = plt.subplots(6, 2, figsize=(7, 14))
for row, (_, pair) in enumerate(show.iterrows()):
    for col, p in enumerate([pair["a"], pair["b"]]):
        axes[row, col].imshow(Image.open(BASE / "image" / p))
        axes[row, col].set_title(info.loc[p, "model_name"], fontsize=9)
        axes[row, col].axis("off")
fig.suptitle("Different car models of the same make, type and viewpoint")
save(fig, "similar_models.png")

# Opsummering
print(f"\nFærdig på {time.time() - start:.0f} s. Figurer gemt i {OUT}:")
for f in sorted(OUT.glob("*.png")):
    print(" ", f.name)
