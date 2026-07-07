from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import yaml
import json
import csv
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from tqdm.auto import tqdm  # progress bars

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
DATE_RANGES_PATH = "conf/date_ranges/default.yaml"

DEVELOPED_ROOTS = [
    "/mnt/research-projects/s/screberg/longterm_images/semifield-developed-images",
    "/mnt/research-projects/s/screberg/GROW_DATA/semifield-developed-images",
    "/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images",
]

UPLOAD_ROOTS = [
    "/mnt/research-projects/s/screberg/longterm_images2/semifield-upload",
]

# Start dates per state (inclusive)
STATE_START_DATES = {
    "NC": "2024-12-02",
    "MD": "2024-04-16",
    "TX": "2025-06-09",
}

TEXT_OUTPUT_PATH = "bbotv31_batches_summary.txt"
JSON_OUTPUT_PATH = "bbotv31_batches.json"

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# --------------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------------

@dataclass(frozen=True)
class Batch:
    state: str
    name: str
    date: str
    image_count: int
    upload_only: bool  # True if found only in UPLOAD_ROOTS
    lts: str           # Long-term storage tag
    has_upload: bool   # True if an upload_dir exists for this batch


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def load_date_ranges(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_batch_date(name: str) -> str | None:
    """Extract YYYY-MM-DD date from a batch folder name."""
    m = DATE_RE.search(name)
    return m.group(1) if m else None


def count_images(image_dir: Path) -> int:
    """
    Count images in a directory (recursively).
    Adjust extensions as needed.
    """
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sum(
        1
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def collect_batches(
    developed_dirs: List[str],
    upload_dirs: List[str],
    state_start_dates: Dict[str, str],
) -> List[Batch]:

    # Mapping of root directory → LTS tag
    LTS_MAP = {
        "/mnt/research-projects/s/screberg/longterm_images/semifield-developed-images": "longterm_images",
        "/mnt/research-projects/s/screberg/GROW_DATA/semifield-developed-images": "GROW_DATA",
        "/mnt/research-projects/s/screberg/longterm_images2/semifield-developed-images": "longterm_images2",
        "/mnt/research-projects/s/screberg/longterm_images2/semifield-upload": "longterm_images2",
    }

    # Map (state, name) -> {"developed": [(Path, lts)], "upload": [(Path, lts)]}
    batch_locations: Dict[Tuple[str, str], Dict[str, List[Tuple[Path, str]]]] = defaultdict(
        lambda: {"developed": [], "upload": []}
    )

    # --------------------------------------
    # Scan developed dirs
    # --------------------------------------
    for droot in developed_dirs:
        base = Path(droot)
        lts_tag = LTS_MAP[droot]
        batch_paths = sorted(base.glob("*"))

        if not batch_paths:
            tqdm.write(f"[INFO] No batch folders found in {droot}")
            continue

        for batch_path in tqdm(
            batch_paths,
            desc=f"Scanning developed batches in {base}",
            unit="batch",
            leave=False,
        ):
            name = batch_path.name
            state = name[:2]
            key = (state, name)
            batch_locations[key]["developed"].append((batch_path, lts_tag))

    # --------------------------------------
    # Scan upload dirs
    # --------------------------------------
    for uroot in upload_dirs:
        base = Path(uroot)
        lts_tag = LTS_MAP[uroot]
        batch_paths = sorted(base.glob("*"))

        if not batch_paths:
            tqdm.write(f"[INFO] No batch folders found in {uroot}")
            continue

        for batch_path in tqdm(
            batch_paths,
            desc=f"Scanning upload batches in {base}",
            unit="batch",
            leave=False,
        ):
            name = batch_path.name
            state = name[:2]
            key = (state, name)
            batch_locations[key]["upload"].append((batch_path, lts_tag))

    batches: List[Batch] = []

    # --------------------------------------
    # Build Batch objects
    # --------------------------------------
    for (state, name), locations in tqdm(
        sorted(batch_locations.items()),
        desc="Collecting batch metadata & counting images",
        unit="batch",
    ):
        if state not in state_start_dates:
            continue

        batch_date = extract_batch_date(name)
        if batch_date is None:
            continue

        if batch_date < state_start_dates[state]:
            continue

        has_dev = len(locations["developed"]) > 0
        has_up = len(locations["upload"]) > 0

        if not has_dev and not has_up:
            continue

        upload_only = (has_up and not has_dev)

        # Choose preferred source and assign correct LTS tag
        if has_dev:
            chosen_path, lts_tag = locations["developed"][0]
            source_label = "developed"
        else:
            chosen_path, lts_tag = locations["upload"][0]
            source_label = "upload"

        # Upload-only → count .RAW files
        if upload_only:
            raw_count = len([
                p for p in chosen_path.glob("*.RAW")
                if p.is_file()
            ])
            img_count = raw_count
            tqdm.write(
                f"[FOUND] {state} {name} (UPLOAD_ONLY, {raw_count} RAW files, LTS={lts_tag})"
            )

        else:
            # Developed or dual → count processed images inside images/
            image_dir = chosen_path / "images"
            if image_dir.exists():
                img_count = count_images(image_dir)
                tqdm.write(
                    f"[FOUND] {state} {name} ({source_label}, images/: {img_count} files, LTS={lts_tag})"
                )
            else:
                img_count = 0
                tqdm.write(
                    f"[FOUND] {state} {name} ({source_label}, NO images/ dir, LTS={lts_tag})"
                )

        batches.append(
            Batch(
                state=state,
                name=name,
                date=batch_date,
                image_count=img_count,
                upload_only=upload_only,
                lts=lts_tag,
                has_upload=has_up,  # NEW: whether upload_dir exists
            )
        )

    # Sorting unchanged
    return sorted(batches, key=lambda b: (b.state, b.name))


def group_batches_by_state_and_season(
    batches: List[Batch],
    date_ranges: Dict,
    states: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Group batches by state and season, returning:
    {
      "NC": {
        "seasons": {season_name: [batch_name, ...], ...},
        "unassigned": [batch_name, ...],
        "batch_seasons": {batch_name: [season1, season2, ...]}
      },
      "MD": { ... },
    }

    NOTE: A batch can belong to multiple seasons if date ranges overlap.
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    for state in states:
        state_date_ranges = date_ranges.get(state, {})
        season_batches: Dict[str, List[str]] = {season: [] for season in state_date_ranges}
        unassigned: List[str] = []
        batch_seasons: Dict[str, List[str]] = {}

        for batch in tqdm(
            batches,
            desc=f"Assigning batches to seasons for {state}",
            unit="batch",
            leave=False,
        ):
            if batch.state != state:
                continue

            membership: List[str] = []
            for season, s_items in state_date_ranges.items():
                if s_items["start"] <= batch.date <= s_items["end"]:
                    season_batches[season].append(batch.name)
                    membership.append(season)

            if membership:
                batch_seasons[batch.name] = membership
            else:
                unassigned.append(batch.name)

        grouped[state] = {
            "seasons": season_batches,
            "unassigned": unassigned,
            "batch_seasons": batch_seasons,
        }

    return grouped


def build_summary_dict(
    grouped: Dict[str, Dict[str, Any]],
    date_ranges: Dict,
    states: List[str],
    batches: List[Batch],
) -> Dict[str, Any]:
    """
    Build a JSON-friendly summary structure from grouped results + date ranges
    + full batch info (including image counts).

    Each batch entry includes:
      - name
      - date
      - image_count
      - seasons: list of seasons this batch belongs to (can be >1)
      - multi_season: bool (True if belongs to more than one season)
      - upload_only: bool (True if batch only exists in upload roots)
      - has_upload: bool (True if upload_dir exists for this batch)
    """
    # Fast lookup from (state, name) -> Batch
    batch_index: Dict[tuple[str, str], Batch] = {
        (b.state, b.name): b for b in batches
    }

    overall_season_counts: Dict[str, int] = {}
    overall_unassigned = 0
    overall_total = 0
    overall_upload_only = 0  # new overall count
    overall_has_upload = 0  # new overall count

    states_summary: Dict[str, Any] = {}

    for state in states:
        state_date_ranges = date_ranges.get(state, {})
        seasons = grouped[state]["seasons"]
        unassigned_names = grouped[state]["unassigned"]
        batch_seasons = grouped[state]["batch_seasons"]

        state_seasons_summary: Dict[str, Any] = {}
        state_total = 0
        state_upload_only = 0
        state_has_upload = 0

        # Per-season
        for season, batch_names in seasons.items():
            s_start = state_date_ranges[season]["start"]
            s_end = state_date_ranges[season]["end"]

            season_batch_entries = []
            for name in sorted(batch_names):
                b = batch_index.get((state, name))
                if b is None:
                    continue
                memberships = batch_seasons.get(name, [])
                multi = len(memberships) > 1
                upload_only = b.upload_only
                has_upload = b.has_upload
                if upload_only:
                    state_upload_only += 1
                    overall_upload_only += 1
                if has_upload:
                    state_has_upload += 1
                    overall_has_upload += 1
                season_batch_entries.append(
                    {
                        "name": b.name,
                        "date": b.date,
                        "image_count": b.image_count,
                        "seasons": memberships,
                        "multi_season": multi,
                        "upload_only": upload_only,
                        "has_upload": has_upload,  # NEW
                        "lts": b.lts,
                    }
                )

            count = len(season_batch_entries)

            state_seasons_summary[season] = {
                "start": s_start,
                "end": s_end,
                "batches": season_batch_entries,
                "count": count,
            }

            state_total += count
            overall_season_counts[season] = overall_season_counts.get(season, 0) + count

        # Unassigned
        unassigned_entries = []
        for name in sorted(unassigned_names):
            b = batch_index.get((state, name))
            if b is None:
                continue
            upload_only = b.upload_only
            has_upload = b.has_upload
            if upload_only:
                state_upload_only += 1
                overall_upload_only += 1
            if has_upload:
                state_has_upload += 1
                overall_has_upload += 1

            # Unassigned has no seasons by definition
            unassigned_entries.append(
                {
                    "name": b.name,
                    "date": b.date,
                    "image_count": b.image_count,
                    "seasons": [],
                    "multi_season": False,
                    "upload_only": upload_only,
                    "has_upload": has_upload,  # NEW
                }
            )

        unassigned_count = len(unassigned_entries)
        state_total += unassigned_count
        overall_unassigned += unassigned_count
        overall_total += state_total

        states_summary[state] = {
            "seasons": state_seasons_summary,
            "unassigned": {
                "batches": unassigned_entries,
                "count": unassigned_count,
            },
            "total_batches": state_total,
            "upload_only_batches": state_upload_only,  # per-state count
            "has_upload_batches": state_has_upload,  # per-state count
        }

    summary = {
        "states": states_summary,
        "summary": {
            "per_season_across_states": overall_season_counts,
            "unassigned_across_states": overall_unassigned,
            "overall_total_batches": overall_total,
            "upload_only_across_states": overall_upload_only,  # new overall summary
            "has_upload_across_states": overall_has_upload,  # new overall summary
        },
    }
    return summary


def print_grouped_to_console(
    grouped: Dict[str, Dict[str, Any]],
    date_ranges: Dict,
    states: List[str],
) -> None:
    """Pretty-print the grouped batches to stdout, including multi-season batches."""
    for state in states:
        print(f"\n{state} Batches and Date Ranges:")
        state_date_ranges = date_ranges.get(state, {})
        seasons = grouped[state]["seasons"]
        unassigned = grouped[state]["unassigned"]
        batch_seasons = grouped[state]["batch_seasons"]

        for season, batch_names in seasons.items():
            s_start = state_date_ranges[season]["start"]
            s_end = state_date_ranges[season]["end"]
            print(f"  {season} ({s_start} -> {s_end}):")
            for b in sorted(batch_names):
                print(f"    {b}")

        if unassigned:
            print("  Unassigned batches:")
            for b in sorted(unassigned):
                print(f"    {b}")

        # Multi-season report
        multi = [b for b, seasons_list in batch_seasons.items() if len(seasons_list) > 1]
        if multi:
            print("  Multi-season batches (overlap detected):")
            for b in sorted(multi):
                seasons_list = ", ".join(sorted(batch_seasons[b]))
                print(f"    {b}  [seasons: {seasons_list}]")


def write_summary_text(
    output_path: str,
    summary: Dict[str, Any],
) -> None:
    """
    Write a human-readable text summary using the summary dict.
    Text stays simple (names), but includes a list of multi-season batches per state.
    Upload-only batches are annotated with [UPLOAD_ONLY].
    Batches that have an upload_dir are annotated with [HAS_UPLOAD].
    """
    states = summary["states"]
    per_season = summary["summary"]["per_season_across_states"]
    overall_unassigned = summary["summary"]["unassigned_across_states"]
    overall_total = summary["summary"]["overall_total_batches"]
    overall_upload_only = summary["summary"]["upload_only_across_states"]
    overall_has_upload = summary["summary"]["has_upload_across_states"]

    with open(output_path, "w") as out_file:
        # Per-state sections
        for state, s_info in states.items():
            out_file.write(f"{state} Batches and Date Ranges:\n")

            # Seasons
            for season, s_season_info in s_info["seasons"].items():
                s_start = s_season_info["start"]
                s_end = s_season_info["end"]
                out_file.write(f"  {season} ({s_start} -> {s_end}):\n")
                for b in s_season_info["batches"]:
                    label = b["name"]
                    tags: List[str] = []
                    if b.get("upload_only"):
                        tags.append("UPLOAD_ONLY")
                    if b.get("has_upload"):
                        tags.append("HAS_UPLOAD")
                    if tags:
                        label += " [" + ", ".join(tags) + "]"
                    out_file.write(f"    {label}\n")

            # Unassigned
            unassigned = s_info["unassigned"]
            if unassigned["batches"]:
                out_file.write("  Unassigned batches:\n")
                for b in unassigned["batches"]:
                    label = b["name"]
                    tags: List[str] = []
                    if b.get("upload_only"):
                        tags.append("UPLOAD_ONLY")
                    if b.get("has_upload"):
                        tags.append("HAS_UPLOAD")
                    if tags:
                        label += " [" + ", ".join(tags) + "]"
                    out_file.write(f"    {label}\n")

            # Multi-season batches for this state
            multi = [
                b
                for b in s_info["unassigned"]["batches"]
                + [bb for s in s_info["seasons"].values() for bb in s["batches"]]
                if b["multi_season"]
            ]
            if multi:
                out_file.write("  Multi-season batches (overlap detected):\n")
                for b in multi:
                    seasons_list = ", ".join(sorted(b["seasons"]))
                    label = b["name"]
                    tags: List[str] = []
                    if b.get("upload_only"):
                        tags.append("UPLOAD_ONLY")
                    if b.get("has_upload"):
                        tags.append("HAS_UPLOAD")
                    if tags:
                        label += " [" + ", ".join(tags) + "]"
                    out_file.write(f"    {label}  [seasons: {seasons_list}]\n")

            out_file.write(f"Total batches for {state}: {s_info['total_batches']}\n")
            out_file.write(
                f"Upload-only batches for {state}: {s_info['upload_only_batches']}\n\n"
            )

        # Overall summaries
        out_file.write("Summary across both states (per season):\n")
        for season in sorted(per_season.keys()):
            out_file.write(f"  {season}: {per_season[season]} batches\n")

        out_file.write(f"  Unassigned (both states): {overall_unassigned} batches\n")
        out_file.write(f"  Upload-only (both states): {overall_upload_only} batches\n")
        out_file.write(f"  Has upload (both states): {overall_has_upload} batches\n")
        out_file.write(f"\nOverall total batches: {overall_total}\n")


def write_summary_json(
    output_path: str,
    summary: Dict[str, Any],
) -> None:
    """Write the same summary structure to JSON."""
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main() -> None:
    date_ranges = load_date_ranges(DATE_RANGES_PATH)
    states = sorted(STATE_START_DATES.keys())

    batches = collect_batches(DEVELOPED_ROOTS, UPLOAD_ROOTS, STATE_START_DATES)
    grouped = group_batches_by_state_and_season(batches, date_ranges, states)

    # Console preview (includes multi-season info)
    print_grouped_to_console(grouped, date_ranges, states)

    # Build reusable summary (includes image_count + season overlap per batch)
    summary = build_summary_dict(grouped, date_ranges, states, batches)

    # Write text + JSON
    write_summary_text(TEXT_OUTPUT_PATH, summary)
    write_summary_json(JSON_OUTPUT_PATH, summary)

    print(f"\nWrote text summary to {TEXT_OUTPUT_PATH}")
    print(f"Wrote JSON summary to {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
