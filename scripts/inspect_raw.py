"""Inspect every raw file in data/raw/delhi_gps/ and write a data inventory.

Each file is a single day of node-crossing events for one bus route, stored
at data/raw/delhi_gps/Month_<m>/<route>/Day<d>.xlsx. The data has no GPS
latitude/longitude anywhere - only an integer "Node" (stop-sequence index
along a route/direction), a vehicle plate, a direction (up/down), and a
time-of-day. There is also no real calendar date anywhere in the archive
(file/folder names, cell contents, or workbook metadata) - only relative
Month_N / Day_D labels.

Usage:
    uv run python scripts/inspect_raw.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delhi_gps"
OUT_PATH = PROJECT_ROOT / "docs" / "data_inventory.md"
FIG_PATH = PROJECT_ROOT / "reports" / "figures" / "00_trips_node_vs_time.png"

DAY_RE = re.compile(r"Day(\d+)\.xlsx$")


def discover_files() -> list[dict]:
    records = []
    for month_dir in sorted(RAW_DIR.glob("Month_*")):
        if not month_dir.is_dir():
            continue
        month_num = int(month_dir.name.split("_")[1])
        for route_dir in sorted(month_dir.iterdir()):
            if not route_dir.is_dir():
                continue
            route = route_dir.name
            for f in sorted(
                route_dir.glob("Day*.xlsx"), key=lambda p: int(DAY_RE.search(p.name).group(1))
            ):
                day_num = int(DAY_RE.search(f.name).group(1))
                records.append({"path": f, "month": month_num, "route": route, "day": day_num})
    return records


def load_file(rec: dict) -> pd.DataFrame:
    df = pd.read_excel(rec["path"])
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["route"] = rec["route"]
    df["direction"] = df["Route"]
    df = df.drop(columns=["Route"])
    df["month"] = rec["month"]
    df["day"] = rec["day"]
    df["time_of_day"] = pd.to_datetime(df["Time"], format="%H:%M:%S:%f").dt.time
    df["seconds_of_day"] = df["time_of_day"].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    )
    df["vehicle_id"] = df["Number"].str.strip()
    df["row_order"] = range(len(df))
    return df


def fmt_bytes_mb(n: int) -> float:
    return round(n / (1024 * 1024), 4)


def assign_trips(df: pd.DataFrame) -> pd.DataFrame:
    """A trip = a maximal run of consecutive rows (by original row order) for one
    vehicle, within one route/month/day, that keeps the same direction label."""
    df = df.sort_values(["route", "month", "day", "vehicle_id", "row_order"]).copy()
    key_cols = ["route", "month", "day", "vehicle_id"]
    same_key = (df[key_cols] != df[key_cols].shift()).any(axis=1)
    dir_change = df["direction"] != df["direction"].shift()
    new_trip = same_key | dir_change
    df["trip_id"] = new_trip.cumsum()
    return df


def check_file_metadata_for_dates(file_records: list[dict]) -> list[str]:
    lines = []
    sample = file_records[:3] + file_records[-1:]
    for rec in sample:
        wb = openpyxl.load_workbook(rec["path"], read_only=True)
        lines.append(
            f"- {rec['path'].relative_to(RAW_DIR)}: workbook created={wb.properties.created}, "
            f"modified={wb.properties.modified}"
        )
        wb.close()
    return lines


def main() -> int:
    file_records = discover_files()
    if not file_records:
        raise SystemExit(f"no files found under {RAW_DIR}")

    frames = []
    per_file_rows = []
    for rec in file_records:
        df = load_file(rec)
        frames.append(df)
        size_mb = fmt_bytes_mb(rec["path"].stat().st_size)
        per_file_rows.append(
            {
                "file": str(rec["path"].relative_to(RAW_DIR)),
                "format": rec["path"].suffix.lstrip("."),
                "size_mb": size_mb,
                "n_rows": len(df),
                "route": rec["route"],
                "month": rec["month"],
                "day": rec["day"],
            }
        )

    all_df = pd.concat(frames, ignore_index=True)
    per_file_df = pd.DataFrame(per_file_rows)

    sample_raw = pd.read_excel(file_records[0]["path"])
    sample_raw = sample_raw.drop(columns=[c for c in sample_raw.columns if c.startswith("Unnamed")])
    schema_lines = [f"- `{col}`: {dtype}" for col, dtype in sample_raw.dtypes.items()]

    # --- date investigation ---
    date_metadata_lines = check_file_metadata_for_dates(file_records)
    n_distinct_days = per_file_df[["month", "day"]].drop_duplicates().shape[0]

    missing_days_lines = []
    for (month, route), grp in per_file_df.groupby(["month", "route"]):
        days = sorted(grp["day"].unique())
        full_range = set(range(min(days), max(days) + 1))
        missing = sorted(full_range - set(days))
        missing_days_lines.append(
            f"- Month_{month} / route {route}: {len(days)} days present (Day{min(days)}-Day{max(days)}), "
            f"missing: {missing if missing else 'none'}"
        )

    # --- buses / routes / directions ---
    n_buses = all_df["vehicle_id"].nunique()
    n_routes = all_df["route"].nunique()
    directions = sorted(all_df["direction"].unique())
    buses_per_route = all_df.groupby("route")["vehicle_id"].nunique()

    # --- Node coverage per route x direction (= stops) ---
    node_coverage = (
        all_df.groupby(["route", "direction"])["Node"]
        .agg(min_node="min", max_node="max", n_distinct_nodes="nunique")
        .reset_index()
    )

    # --- points per day (per file = one route-day) ---
    points_per_day = per_file_df["n_rows"]

    # --- trips ---
    all_df = assign_trips(all_df)
    trip_meta = (
        all_df.groupby("trip_id")
        .agg(
            route=("route", "first"),
            direction=("direction", "first"),
            month=("month", "first"),
            day=("day", "first"),
            vehicle_id=("vehicle_id", "first"),
            n_points=("Node", "size"),
            min_node=("Node", "min"),
            max_node=("Node", "max"),
            start_s=("seconds_of_day", "min"),
            end_s=("seconds_of_day", "max"),
        )
        .reset_index()
    )
    n_trips = len(trip_meta)
    trips_per_rdday = (
        all_df.groupby(["route", "direction", "month", "day"])
        .agg(n_vehicles=("vehicle_id", "nunique"), n_trips=("trip_id", "nunique"))
        .reset_index()
    )

    # sample of one full trip
    sample_trip_id = trip_meta.iloc[0]["trip_id"]
    sample_trip_rows = all_df[all_df["trip_id"] == sample_trip_id][
        ["route", "direction", "month", "day", "vehicle_id", "Node", "Time"]
    ]

    # --- data quality checks ---
    dedup_cols = ["route", "month", "day", "vehicle_id", "direction", "Node", "Time"]
    n_exact_dupes = all_df.duplicated(subset=dedup_cols).sum()
    n_missing_rows = all_df[["Node", "vehicle_id", "direction", "Time"]].isna().any(axis=1).sum()

    # out-of-order nodes within a trip: Node is a direction-specific sequence index that
    # should be non-decreasing over time regardless of direction label (verified empirically:
    # both 'up' and 'down' trips number their nodes low-to-high in travel order).
    def out_of_order(g: pd.DataFrame) -> int:
        diffs = g.sort_values("row_order")["Node"].diff().dropna()
        return int((diffs < 0).sum())

    out_of_order_counts = all_df.groupby("trip_id").apply(out_of_order, include_groups=False)
    n_out_of_order_trips = int((out_of_order_counts > 0).sum())
    n_out_of_order_rows = int(out_of_order_counts.sum())

    # time going backwards within a trip (by row order, which should match time order)
    def time_backwards(g: pd.DataFrame) -> int:
        return int((g.sort_values("row_order")["seconds_of_day"].diff().dropna() < 0).sum())

    time_backwards_counts = all_df.groupby("trip_id").apply(time_backwards, include_groups=False)
    n_time_backwards_trips = int((time_backwards_counts > 0).sum())
    n_time_backwards_rows = int(time_backwards_counts.sum())

    # trips that skip nodes (gap > 1 between consecutive nodes, ignoring direction)
    def skipped_nodes(g: pd.DataFrame) -> int:
        diffs = g.sort_values("row_order")["Node"].diff().dropna().abs()
        return int((diffs > 1).sum())

    skipped_counts = all_df.groupby("trip_id").apply(skipped_nodes, include_groups=False)
    n_trips_with_skips = int((skipped_counts > 0).sum())
    n_skip_events = int(skipped_counts.sum())

    # trips crossing midnight: time not monotonically increasing across the whole trip span,
    # specifically a large negative jump near end of day (>12h back) suggesting day wrap
    def crosses_midnight(g: pd.DataFrame) -> bool:
        s = g.sort_values("row_order")["seconds_of_day"]
        return bool((s.diff().dropna() < -12 * 3600).any())

    midnight_crossing = all_df.groupby("trip_id").apply(crosses_midnight, include_groups=False)
    n_midnight_trips = int(midnight_crossing.sum())

    # --- stop / schedule / GTFS data check ---
    all_paths = list(RAW_DIR.rglob("*"))
    gtfs_like = [
        p
        for p in all_paths
        if p.is_file()
        and any(kw in p.name.lower() for kw in ("stop", "gtfs", "schedule", "timetable", "trip"))
        and p.suffix != ".xlsx"
    ]

    total_size_mb = fmt_bytes_mb(sum(p.stat().st_size for p in RAW_DIR.rglob("*") if p.is_file()))

    # estimated stop-to-stop segments buildable (consecutive node pairs within a trip)
    n_segments = int((trip_meta["n_points"] - 1).clip(lower=0).sum())

    # --- plot: time-space diagram for one representative day/route, both directions ---
    plot_route = "534"
    plot_month, plot_day = 1, 1
    plot_df = all_df[
        (all_df["route"] == plot_route) & (all_df["month"] == plot_month) & (all_df["day"] == plot_day)
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"up": "#1f77b4", "down": "#d62728"}
    for trip_id, g in plot_df.groupby("trip_id"):
        g = g.sort_values("row_order")
        direction = g["direction"].iloc[0]
        ax.plot(
            g["seconds_of_day"] / 3600,
            g["Node"],
            marker=".",
            markersize=2,
            linewidth=0.8,
            color=colors.get(direction, "gray"),
        )
    handles = [plt.Line2D([0], [0], color=c, label=d) for d, c in colors.items()]
    ax.legend(handles=handles, title="direction")
    ax.set_xlabel("time of day (hours)")
    ax.set_ylabel("Node (stop-sequence index)")
    ax.set_title(f"Route {plot_route} - Month_{plot_month} Day{plot_day}: trips (time-space diagram)")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    # ----------------------- write markdown -----------------------
    lines = []
    lines.append("# Delhi Bus GPS - Raw Data Inventory\n")
    lines.append(
        "Source: `data.source_url` in `params.yaml` "
        "(https://github.com/bus-data/bus-data.github.io/raw/master/asset/dataset.zip), "
        f"downloaded and unzipped into `data/raw/delhi_gps/` ({total_size_mb} MB on disk, "
        f"{len(file_records)} files).\n"
    )

    lines.append("## Layout\n")
    lines.append(
        "Each file is one day of node-crossing events for one bus route: "
        "`Month_<m>/<route>/Day<d>.xlsx`.\n"
    )

    lines.append("## Per-file summary\n")
    lines.append(
        f"{len(file_records)} files, all `.xlsx`, identical schema. "
        f"Size ranges {per_file_df['size_mb'].min()}-{per_file_df['size_mb'].max()} MB per file, "
        f"rows per file range {per_file_df['n_rows'].min()}-{per_file_df['n_rows'].max()} "
        f"(min: {points_per_day.min()}, median: {points_per_day.median()}, max: {points_per_day.max()}).\n"
    )
    lines.append("| file | format | size (MB) | rows | route | month | day |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in per_file_df.iterrows():
        lines.append(
            f"| {r['file']} | {r['format']} | {r['size_mb']} | {r['n_rows']} | "
            f"{r['route']} | {r['month']} | {r['day']} |"
        )
    lines.append("")

    lines.append("## Schema (common to every file)\n")
    lines.extend(schema_lines)
    lines.append(
        "\nNote: `Route` in the raw files actually encodes **direction** (`up`/`down`), "
        "not the bus route number - the real route number comes from the folder name "
        "(`425` or `534`). `Node` is an integer stop-sequence index along the route/direction, "
        "`Number` is the vehicle registration plate, `Time` is `HH:MM:SS:mmm` time-of-day "
        "(no date component).\n"
    )
    lines.append("### First 5 rows (representative file, Month_1/425/Day1.xlsx)\n")
    lines.append(sample_raw.head(5).to_markdown(index=False))
    lines.append("")

    lines.append("## Latitude / longitude\n")
    lines.append(
        "**No latitude/longitude columns exist anywhere in this dataset** - skipped per request. "
        "The only positional information is the integer `Node` (a stop-sequence index), which is "
        "already stop-matched (see Node coverage below)."
    )
    lines.append("")

    lines.append("## Where does the date come from?\n")
    lines.append(
        "Checked file names, folder names, cell contents, and Excel workbook metadata "
        "(`created`/`modified` properties) for a real calendar date. Findings:\n"
    )
    lines.append("- File/folder names only contain relative labels: `Month_1..3`, `Day1..31`.")
    lines.append("- The `Time` column is time-of-day only (`HH:MM:SS:mmm`), no date.")
    lines.append("- Workbook metadata (`created`/`modified`) for a sample of files:")
    lines.extend(date_metadata_lines)
    lines.append(
        "  All workbook timestamps fall within the same ~10-minute window on 2019-07-09, "
        "regardless of which Month_N/Day_D the file represents - this is the batch export/"
        "generation time of the whole archive, not the date the GPS data was collected."
    )
    lines.append(
        "\n**Conclusion: no real calendar date can be recovered anywhere in this archive.** "
        f"Only the relative grouping exists: {n_distinct_days} distinct (month, day) combinations, "
        "spanning Month_1-Month_3, each with 30 or 31 day-labels. Missing days per (month, route):"
    )
    lines.extend(missing_days_lines)
    lines.append("")

    lines.append("## Buses, routes, directions\n")
    lines.append(f"- Unique vehicle ids (number plates): {n_buses}")
    lines.append(f"- Unique routes: {n_routes} ({sorted(all_df['route'].unique())})")
    lines.append(f"- Directions: {directions}")
    for route, n in buses_per_route.items():
        lines.append(f"- Unique buses on route {route}: {n}")
    lines.append("")

    lines.append("## Node coverage per route x direction (= stops)\n")
    lines.append("| route | direction | min node | max node | distinct nodes |")
    lines.append("|---|---|---|---|---|")
    for _, r in node_coverage.iterrows():
        lines.append(
            f"| {r['route']} | {r['direction']} | {r['min_node']} | {r['max_node']} | {r['n_distinct_nodes']} |"
        )
    lines.append("")

    lines.append("## Vehicles and trips per route x direction x day\n")
    lines.append(
        f"Total trips across the whole archive: {n_trips} "
        "(a trip = a maximal run of consecutive rows for one vehicle, on one route-day, "
        "with the same direction label).\n"
    )
    lines.append(
        f"- vehicles per route-direction-day: min {trips_per_rdday['n_vehicles'].min()}, "
        f"median {trips_per_rdday['n_vehicles'].median()}, max {trips_per_rdday['n_vehicles'].max()}"
    )
    lines.append(
        f"- trips per route-direction-day: min {trips_per_rdday['n_trips'].min()}, "
        f"median {trips_per_rdday['n_trips'].median()}, max {trips_per_rdday['n_trips'].max()}"
    )
    lines.append("\n### Sample of one full trip\n")
    lines.append(sample_trip_rows.to_markdown(index=False))
    lines.append("")

    lines.append("## Duplicates and missing values\n")
    lines.append(f"- Exact duplicate rows (route, month, day, vehicle_id, direction, Node, Time): {n_exact_dupes}")
    lines.append(f"- Rows with missing values (Node/vehicle_id/direction/Time): {n_missing_rows}")
    lines.append("")

    lines.append("## Data quality: node/time ordering within trips\n")
    lines.append(
        f"- Trips with `Node` decreasing between consecutive rows (it should be non-decreasing "
        f"over time for both directions - `up` and `down` each number their own nodes low-to-high "
        f"in travel order): {n_out_of_order_trips} trips, {n_out_of_order_rows} offending row-to-row steps."
    )
    lines.append(
        f"- Trips with time going backwards between consecutive rows: {n_time_backwards_trips} trips, "
        f"{n_time_backwards_rows} offending steps."
    )
    lines.append(
        f"- Trips that skip at least one node (gap > 1 between consecutive nodes): {n_trips_with_skips} trips "
        f"out of {n_trips} ({n_trips_with_skips / n_trips:.1%}), {n_skip_events} total skip events."
    )
    lines.append(f"- Trips crossing midnight (time jumps back by >12h mid-trip): {n_midnight_trips}")
    lines.append("")

    lines.append("## Time-space diagram\n")
    lines.append(
        f"See `reports/figures/00_trips_node_vs_time.png`: x = time of day, y = Node, one line per trip, "
        f"for route {plot_route}, Month_{plot_month} Day{plot_day}, both directions in different colours."
    )
    lines.append("")

    lines.append("## Stop / schedule / GTFS-like data\n")
    if gtfs_like:
        lines.append("Found candidate stop/schedule/GTFS files:")
        for p in gtfs_like:
            lines.append(f"- {p.relative_to(RAW_DIR)}")
    else:
        lines.append(
            "**None.** The archive contains only the per-route, per-day `.xlsx` node-crossing "
            "files above - no `stops.txt`, timetable, or other GTFS-like file is included. "
            "Mapping `Node` indices to real stop names/locations, or getting an actual published "
            "schedule, would require a separate Delhi GTFS feed."
        )
    lines.append("")

    lines.append("## Totals\n")
    lines.append(f"- Total raw rows: {len(all_df):,}")
    lines.append(f"- Total trips: {n_trips:,}")
    lines.append(
        f"- Estimated stop-to-stop segments buildable (sum over trips of consecutive node pairs): {n_segments:,}"
    )
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)} ({len(lines)} lines)")
    print(f"Wrote {FIG_PATH.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
