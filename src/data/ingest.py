"""Download and unzip the raw Delhi bus GPS dataset.

Downloads the zip archive at `data.source_url` (in params.yaml) into
data/raw/delhi_gps/ and extracts it there. Skips the download/extraction
if the files already exist.

Usage:
    uv run python -m src.data.ingest
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delhi_gps"
ZIP_PATH = RAW_DIR / "dataset.zip"

TIMEOUT = 60
CHUNK_SIZE = 1 << 16  # 64 KiB


class IngestError(RuntimeError):
    pass


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size = dest.stat().st_size
        print(f"  SKIP  {dest.relative_to(PROJECT_ROOT)} already exists ({size:,} bytes)")
        return

    try:
        with requests.get(url, timeout=TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            tmp_path = dest.with_suffix(dest.suffix + ".part")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            tmp_path.rename(dest)
    except requests.RequestException as exc:
        raise IngestError(f"failed to download {url}: {exc}") from exc

    size = dest.stat().st_size
    print(f"  OK    {dest.relative_to(PROJECT_ROOT)} ({size:,} bytes)")


def extract(zip_path: Path, dest_dir: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            already_extracted = all((dest_dir / n).exists() for n in names)
            if already_extracted:
                print(f"  SKIP  extraction, {len(names)} member file(s) already present")
                return
            zf.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        raise IngestError(f"{zip_path} is not a valid zip file: {exc}") from exc

    for name in names:
        path = dest_dir / name
        size = path.stat().st_size
        print(f"  OK    {path.relative_to(PROJECT_ROOT)} ({size:,} bytes)")


def main() -> int:
    params = load_params()
    source_url = params["data"]["source_url"]

    print(f"Downloading {source_url}")
    try:
        download(source_url, ZIP_PATH)
        print("Extracting archive")
        extract(ZIP_PATH, RAW_DIR)
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
