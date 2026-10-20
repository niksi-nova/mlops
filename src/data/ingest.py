"""Download and stage the raw Delhi bus GPS dataset.

Downloads the zip archive at `data.source_url` (in params.yaml) into the
git-ignored data/cache/ folder and extracts it there. Then copies only the
months listed in `data.months_included` into data/raw/delhi_gps/ (the
DVC-tracked raw data folder), preserving the original Month_<m>/<route>/
Day<d>.xlsx layout and removing any month folders that are no longer
included.

Usage:
    uv run python -m src.data.ingest
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "delhi_gps"
ZIP_PATH = CACHE_DIR / "dataset.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delhi_gps"

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


def stage_months(cache_dir: Path, raw_dir: Path, months_included: list[int]) -> None:
    included_names = {f"Month_{m}" for m in months_included}

    raw_dir.mkdir(parents=True, exist_ok=True)

    # remove any raw month folders that are no longer included
    for existing in sorted(raw_dir.glob("Month_*")):
        if existing.is_dir() and existing.name not in included_names:
            shutil.rmtree(existing)
            print(f"  REMOVE {existing.relative_to(PROJECT_ROOT)} (not in months_included)")

    for month_name in sorted(included_names):
        src = cache_dir / month_name
        if not src.is_dir():
            raise IngestError(f"month folder {month_name!r} not found in cache at {src}")
        dest = raw_dir / month_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"  COPY  {src.relative_to(PROJECT_ROOT)} -> {dest.relative_to(PROJECT_ROOT)}")


def main() -> int:
    params = load_params()
    source_url = params["data"]["source_url"]
    months_included = params["data"]["months_included"]

    print(f"Downloading {source_url}")
    try:
        download(source_url, ZIP_PATH)
        print("Extracting archive into cache")
        extract(ZIP_PATH, CACHE_DIR)
        print(f"Staging months {months_included} into {RAW_DIR.relative_to(PROJECT_ROOT)}")
        stage_months(CACHE_DIR, RAW_DIR, months_included)
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
