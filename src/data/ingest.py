"""Download raw TTC bus delay data from the City of Toronto Open Data CKAN API.

Resources are looked up by name via `package_show` on every run - no
resource URLs or ids are hardcoded, since they can change between CKAN
re-publishes.

Usage:
    uv run python -m src.data.ingest
    uv run python -m src.data.ingest --include-recent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

TIMEOUT = 60
CHUNK_SIZE = 1 << 16  # 64 KiB

REFERENCE_RESOURCE_NAME = "Code Descriptions.csv"
RECENT_RESOURCE_NAME = "TTC Bus Delay Data since 2025.csv"


class IngestError(RuntimeError):
    pass


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def fetch_package(ckan_base: str, package_id: str) -> dict:
    url = ckan_base.rstrip("/") + "/package_show"
    try:
        resp = requests.get(url, params={"id": package_id}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise IngestError(f"package_show request failed for {package_id!r}: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError as exc:
        raise IngestError(
            f"package_show returned non-JSON (HTTP {resp.status_code}) for {package_id!r}"
        ) from exc

    if not payload.get("success", False):
        error = payload.get("error", {})
        raise IngestError(
            f"package_show failed for {package_id!r}: "
            f"{error.get('__type', 'Error')} - {error.get('message', error)}"
        )
    return payload["result"]


def find_resource(resources: list[dict], name: str) -> dict:
    matches = [r for r in resources if r.get("name") == name]
    if not matches:
        available = ", ".join(sorted(r.get("name", "") for r in resources))
        raise IngestError(f"resource {name!r} not found in package. Available: {available}")
    if len(matches) > 1:
        raise IngestError(f"resource name {name!r} is ambiguous: {len(matches)} matches found")
    return matches[0]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-recent",
        action="store_true",
        help=f"also download {RECENT_RESOURCE_NAME!r} into data/raw/recent/",
    )
    args = parser.parse_args()

    params = load_params()
    data_cfg = params["data"]
    ckan_base = data_cfg["ckan_base"]
    package_id = data_cfg["package_id"]
    years = data_cfg["historical_years"]

    print(f"Fetching package_show for {package_id!r} ...")
    try:
        package = fetch_package(ckan_base, package_id)
    except IngestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    resources = package.get("resources") or []
    print(f"Package {package_id!r} has {len(resources)} resource(s), state={package.get('state')}\n")

    failures = 0

    print("Historical yearly files -> data/raw/historical/")
    for year in years:
        resource_name = f"ttc-bus-delay-data-{year}"
        try:
            resource = find_resource(resources, resource_name)
            dest = RAW_DIR / "historical" / f"{resource_name}.xlsx"
            download(resource["url"], dest)
        except IngestError as exc:
            print(f"  FAIL  {resource_name}: {exc}", file=sys.stderr)
            failures += 1

    print("\nReference file -> data/raw/reference/")
    try:
        resource = find_resource(resources, REFERENCE_RESOURCE_NAME)
        dest = RAW_DIR / "reference" / REFERENCE_RESOURCE_NAME
        download(resource["url"], dest)
    except IngestError as exc:
        print(f"  FAIL  {REFERENCE_RESOURCE_NAME}: {exc}", file=sys.stderr)
        failures += 1

    if args.include_recent:
        print("\nRecent (since 2025) file -> data/raw/recent/")
        try:
            resource = find_resource(resources, RECENT_RESOURCE_NAME)
            dest = RAW_DIR / "recent" / RECENT_RESOURCE_NAME
            download(resource["url"], dest)
        except IngestError as exc:
            print(f"  FAIL  {RECENT_RESOURCE_NAME}: {exc}", file=sys.stderr)
            failures += 1
    else:
        print("\n(skipping recent data; pass --include-recent to download it)")

    if failures:
        print(f"\n{failures} download(s) failed.", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
