"""Download the pinned benchmark files and verify their upstream SHA-256."""

import argparse
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from src.evidence import file_digest


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "configs/dataset_manifest.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=manifest, nargs="*")
    requested = parser.parse_args().dataset
    for dataset in requested or manifest:
        source = manifest[dataset]
        directory = root / "data" / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for filename, expected in source["files"].items():
            path = directory / filename
            if path.exists() and file_digest(path) == expected["sha256"]:
                print(f"Verified {path}", flush=True)
                continue
            temporary = path.with_suffix(".download")
            url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{filename}"
            print(f"Downloading {filename}", flush=True)
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != expected["bytes"] or file_digest(temporary) != expected["sha256"]:
                raise ValueError(f"Download integrity check failed for {filename}")
            temporary.replace(path)


if __name__ == "__main__":
    main()
