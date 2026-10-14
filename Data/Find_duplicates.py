from pathlib import Path
import hashlib
import time
import pandas as pd

start = time.time()
BASE = Path(__file__).parent
df = pd.read_csv(BASE / "index.csv")

# Et md5-fingeraftryk af hver fil: ens bytes giver samme fingeraftryk
df["md5"] = [hashlib.md5((BASE / "image" / p).read_bytes()).hexdigest() for p in df["path"]]

# Find fingeraftryk der optræder mere end én gang
dups = df[df.duplicated("md5", keep=False)].sort_values("md5")
groups = dups.groupby("md5")
print("Billeder der er dubletter:", len(dups))
print("Grupper af ens billeder:", groups.ngroups)

# Ligger dubletterne under samme model eller på tværs af modeller?
models_per_group = groups["model_id"].nunique()
print("Grupper på tværs af forskellige modeller:", (models_per_group > 1).sum())

# Gem listen, så vi kan kigge på den
dups[["md5", "path", "make_id", "model_id", "year", "viewpoint"]].to_csv(BASE / "duplicates.csv", index=False)
print(f"Færdig på {time.time() - start:.0f} s. Gemt i duplicates.csv")
