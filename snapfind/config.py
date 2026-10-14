from pathlib import Path
import hashlib
import json

import yaml

# stier læses fra repoets rod
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path=None, overrides=()):
    with open(repo_path(path) if path else DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    for item in overrides:
        key, equals, value = item.partition("=")
        if not equals:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        *parents, last = key.split(".")
        node = cfg
        try:
            for name in parents:
                node = node[name]
            node[last]  # fanger tastefejl i --set
        except (KeyError, TypeError):
            raise SystemExit(f"Unknown config key: {key}")
        node[last] = yaml.safe_load(value)
    return cfg


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)
