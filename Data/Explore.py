from pathlib import Path
from collections import Counter
import statistics

#Lavet af Claude Code

root = Path(__file__).parent / "image"
if not root.exists():
    raise SystemExit(f"Could not find {root}")

counts = Counter()  # (make_id, model_id) -> number of images
for img in root.rglob("*.jpg"):
    parts = img.relative_to(root).parts  # (make, model, year, filename)
    if len(parts) == 4:
        counts[(parts[0], parts[1])] += 1

total = sum(counts.values())
makes = {make for make, _ in counts}
per_model = sorted(counts.values())

print("Total images:", total)
print("Makes:", len(makes))
print("Models:", len(counts))
print("Images per model — min:", per_model[0],
      "median:", statistics.median(per_model),
      "max:", per_model[-1])

for t in (20, 40, 80):
    print(f"Models with at least {t} images:", sum(1 for c in per_model if c >= t))