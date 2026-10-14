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

## Repo structure

```
Data/                 scripts for downloading and exploring CompCars
  download_CompCars.py   downloads the dataset from Kaggle
  Build_index.py         makes Data/index.csv (one row per image with label and bounding box)
  Find_duplicates.py     finds identical images, writes Data/duplicates.csv
  Explore.py             quick counts of images per make and model
  Explore_data.py        plots and numbers for the data chapter

snapfind/             the experiment code
  __main__.py            the commands (split, features, evaluate, costs, all)
  config.py              reads configs/default.yaml
  data.py                loads images, crops to the bounding box, transforms
  split.py               splits car models into train / test / validation and picks queries
  models.py              ResNet-18 pretrained on ImageNet
  features.py            runs the network on all images, saves features.npy
  hashing.py             turns features into binary codes (LSH and PCA)
  search.py              FAISS search, float and Hamming
  metrics.py             mAP@10, P@10 and bootstrap intervals
  evaluate.py            runs the search for A and B and writes results and figures
  diagnostics.py         sanity checks (near duplicates, shuffled labels)
  costs.py               encoding time and image size on a device

configs/default.yaml  all settings (seed, split sizes, bits, k)
results/baseline/     results from the full run (CSV, JSON and figures)
baseline.py           same as python -m snapfind
```
