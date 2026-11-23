Gruppemedlemmer:
Osama Awad

# Snap & Find

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Data/download_CompCars.py
```

Link the downloaded folders (image, label, misc, part, train_test_split) into `Data/`, e.g.

```bash
ln -s <path from download>/image Data/image
```

Then build the index:

```bash
python Data/Build_index.py
python Data/Find_duplicates.py
```

## Run

Quick test on a small subset:

```bash
python -m snapfind all --limit 5000 --out results/baseline_test
```

Full run:

```bash
python -m snapfind split
python -m snapfind features
python -m snapfind evaluate
python -m snapfind costs --device-label aau-gpu
```

Phase 2 (trains the network, run on the AAU GPU):

```bash
python -m snapfind train --variant Aft
for BITS in 16 32 64 128; do python -m snapfind train --variant C --bits $BITS; done

python -m snapfind features --checkpoint runs/Aft/best.pt --out results/Aft
for BITS in 16 32 64 128; do
  python -m snapfind features --checkpoint runs/C_$BITS/best.pt --out results/C_$BITS
done

python -m snapfind evaluate --variants A B Aft Bft C --out results/phase2
```

Each training run writes `runs/<run>/` with `best.pt` (best mAP@10 on the unseen validation car models),
`last.pt`, `metrics.csv` and `curves.png`.

## Repo structure

```
Data/                 scripts for downloading and exploring CompCars
  download_CompCars.py   downloads the dataset from Kaggle
  Build_index.py         makes Data/index.csv (one row per image with label and bounding box)
  Find_duplicates.py     finds identical images, writes Data/duplicates.csv
  Explore.py             quick counts of images per make and model
  Explore_data.py        plots and numbers for the data chapter

snapfind/             the experiment code
  __main__.py            the commands (split, features, evaluate, costs, all, train)
  config.py              reads configs/default.yaml
  data.py                loads images, crops to the bounding box, transforms
  split.py               splits car models into train / test / validation and picks queries
  models.py              ResNet-18 pretrained on ImageNet, and the trained network for A_ft and C
  features.py            runs the network on all images, saves features.npy
  hashing.py             turns features into binary codes (LSH and PCA)
  search.py              FAISS search, float and Hamming
  metrics.py             mAP@10, P@10 and bootstrap intervals
  train.py               trains A_ft (fine-tuned float) or C (deep hash)
  evaluate.py            runs the search for A, B, A_ft, B_ft and C and writes results and figures
  diagnostics.py         sanity checks (near duplicates, shuffled labels)
  costs.py               encoding time and image size on a device

configs/default.yaml  all settings (seed, split sizes, bits, k)
results/baseline/     results from the full run (CSV, JSON and figures)
baseline.py           same as python -m snapfind
```
