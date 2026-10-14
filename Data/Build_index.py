from pathlib import Path
import csv
import os
from PIL import Image

data = Path(__file__).parent
image_root = data / "image"
label_root = data / "label"
out_file = data / "index.csv"

columns = ["path", "make_id", "model_id", "year", "viewpoint",
           "x1", "y1", "x2", "y2", "width", "height", "file_bytes"]
rows = []
missing = 0
unreadable = 0

# image/<make_id>/<model_id>/<year>/<navn>.jpg
for img in sorted(image_root.rglob("*.jpg")):
    rel = img.relative_to(image_root)
    make_id, model_id, year, _ = rel.parts
    label = (label_root / rel).with_suffix(".txt")

    # spring over hvis der ikke er en label
    if not label.exists():
        missing += 1
        continue

    try:
        # linje 1: viewpoint, linje 3: bounding box
        lines = label.read_text().split("\n")
        viewpoint = int(lines[0])
        x1, y1, x2, y2 = map(int, lines[2].split())

        with Image.open(img) as im:
            width, height = im.size
    except Exception:
        unreadable += 1
        continue

    rows.append([rel.as_posix(), make_id, model_id, year, viewpoint,
                 x1, y1, x2, y2, width, height, os.path.getsize(img)])

with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(rows)

print("Total rows:", len(rows))
print("Makes:", len({r[1] for r in rows}))
print("Models:", len({(r[1], r[2]) for r in rows}))
print("Missing label file:", missing)
print("Unreadable label/image:", unreadable)
print("Saved to", out_file)
