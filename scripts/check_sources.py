"""List TTC bus delay data sources on the City of Toronto Open Data portal.

Queries the CKAN API and prints matching packages and their resources.
No data files are downloaded.

Usage:
    uv run python scripts/check_sources.py
"""

from __future__ import annotations

import sys

import requests

BASE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/"
SEARCH_QUERY = "ttc bus delay"
DIRECT_PACKAGE_ID = "ttc-bus-delay-data"
TIMEOUT = 30


class CkanError(RuntimeError):
    pass


def ckan_call(action: str, **params) -> dict:
    """Call a CKAN action and return its `result` payload."""
    resp = requests.get(BASE_URL + action, params=params, timeout=TIMEOUT)
    try:
        payload = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise CkanError(f"{action}: non-JSON response (HTTP {resp.status_code})")
    if not payload.get("success", False):
        error = payload.get("error", {})
        raise CkanError(f"{action}: {error.get('__type', 'Error')} - {error.get('message', error)}")
    return payload["result"]


def print_resources(package: dict) -> None:
    resources = package.get("resources") or []
    if not resources:
        print("    (no resources)")
        return
    for res in resources:
        print(f"    - name:          {res.get('name')}")
        print(f"      format:        {res.get('format') or '(none)'}")
        print(f"      last_modified: {res.get('last_modified') or '(none)'}")
        print(f"      url:           {res.get('url')}")


def search_packages() -> list[dict]:
    print(f"=== package_search q={SEARCH_QUERY!r} ===")
    result = ckan_call("package_search", q=SEARCH_QUERY, rows=100)
    packages = result.get("results", [])
    print(f"Found {result.get('count', len(packages))} package(s)\n")
    for pkg in packages:
        print(f"  name:  {pkg.get('name')}")
        print(f"  title: {pkg.get('title')}")
        print(f"  state: {pkg.get('state')}\n")
    return packages


def show_packages(packages: list[dict]) -> None:
    print("=== package_show for each match ===")
    for pkg in packages:
        name = pkg.get("name")
        print(f"\n[{name}]")
        try:
            print_resources(ckan_call("package_show", id=name))
        except (CkanError, requests.RequestException) as exc:
            print(f"    ERROR: {exc}")


def show_direct_package() -> None:
    print(f"\n=== package_show id={DIRECT_PACKAGE_ID!r} ===")
    try:
        pkg = ckan_call("package_show", id=DIRECT_PACKAGE_ID)
    except CkanError as exc:
        print(f"  Package unavailable (likely retired or removed): {exc}")
        return
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return

    print(f"  title: {pkg.get('title')}")
    print(f"  state: {pkg.get('state')}")
    refresh = pkg.get("refresh_rate")
    if refresh:
        print(f"  refresh_rate: {refresh}")
    if pkg.get("state") != "active" or (refresh or "").lower() == "retired":
        print("  NOTE: this package appears to be retired/inactive.")
    if not pkg.get("resources"):
        print("  NOTE: this package has no resources.")
    print_resources(pkg)


def main() -> int:
    try:
        packages = search_packages()
    except (CkanError, requests.RequestException) as exc:
        print(f"package_search failed: {exc}", file=sys.stderr)
        return 1
    show_packages(packages)
    show_direct_package()
    return 0


if __name__ == "__main__":
    sys.exit(main())
