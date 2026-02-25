#!/usr/bin/env python3
"""
Show player course history for the current week's tournament.

For each player in this week's field, displays their finish results
and pre-tournament win probability from past editions of this event
(2020-2025 via the DataGolf archive endpoint).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pga_oad import DataGolfClient

HISTORY_YEARS = [2025, 2024, 2023, 2022, 2021, 2020]

# Finish text sort order for ranking (lower = better)
FINISH_ORDER = {"1": 1, "CUT": 9998, "WD": 9999, "DQ": 9999, "MDF": 9997}


def finish_sort_key(fin_text: str) -> int:
    if not fin_text:
        return 10000
    clean = fin_text.lstrip("T").lstrip("=")
    try:
        return int(clean)
    except ValueError:
        return FINISH_ORDER.get(fin_text, 9000)


def main():
    parser = argparse.ArgumentParser(description="Course history for current week's PGA event")
    parser.add_argument("--top", type=int, default=30, help="Show top N players by current win% (default: 30)")
    parser.add_argument("--player", type=str, default=None, help="Filter to a specific player name (partial match)")
    parser.add_argument("--event-id", type=int, default=None, help="Override event ID (default: auto-detect from current predictions)")
    args = parser.parse_args()

    client = DataGolfClient()

    # Get current event predictions to identify the active event and current field
    print("Fetching current week predictions...")
    current_data = client.get_pre_tournament_predictions(tour="pga", odds_format="percent")
    event_name = current_data.get("event_name", "Unknown Event")
    current_baseline = current_data.get("baseline", [])
    current_history_fit = {p["dg_id"]: p for p in current_data.get("baseline_history_fit", [])}

    if not current_baseline:
        print("No active tournament predictions found.")
        return

    # Build current field lookup: dg_id -> player data
    current_field: dict[int, dict] = {p["dg_id"]: p for p in current_baseline}

    # Determine event_id: use override or auto-detect from schedule
    event_id = args.event_id
    if event_id is None:
        print("Fetching schedule to identify event ID...")
        schedule_data = client.get_schedule(tour="pga", upcoming_only=True)
        for e in schedule_data.get("schedule", []):
            sched_name = e.get("event_name", "").lower()
            if event_name.lower() in sched_name or sched_name in event_name.lower():
                event_id = e.get("event_id")
                break
        if event_id is None and schedule_data.get("schedule"):
            event_id = schedule_data["schedule"][0].get("event_id")

    if event_id is None:
        print("Could not determine event ID. Use --event-id to specify it manually.")
        return

    print(f"\nEvent: {event_name}  (ID: {event_id})")
    print(f"Fetching course history for years: {HISTORY_YEARS}")
    print()

    # Fetch historical archive data for each year
    history: dict[int, dict[int, dict]] = {}  # year -> {dg_id -> player_data}
    for year in HISTORY_YEARS:
        try:
            arch = client.get_pre_tournament_archive(event_id=event_id, year=year)
            players = arch.get("baseline", [])
            history[year] = {p["dg_id"]: p for p in players}
        except Exception as e:
            history[year] = {}

    # Filter current field
    if args.player:
        filter_lower = args.player.lower()
        field_players = [p for p in current_baseline if filter_lower in p["player_name"].lower()]
    else:
        # Sort by win prob descending, take top N
        field_players = sorted(current_baseline, key=lambda p: p.get("win", 0), reverse=True)[:args.top]

    if not field_players:
        print("No players found matching filter.")
        return

    # Determine which years actually had data
    years_with_data = [y for y in HISTORY_YEARS if any(history[y] for y in [y])]

    # Build header
    year_cols = "  ".join(f"{y}" for y in HISTORY_YEARS)
    header = f"{'Rank':<5} {'Player':<28} {'Win%(Base)':>11} {'Win%(+Hist)':>12}   {year_cols}"
    print(header)
    print("-" * (len(header) + 10))

    for rank, p in enumerate(field_players, 1):
        dg_id = p["dg_id"]
        name = p["player_name"]
        win_base = p.get("win", 0.0) * 100
        hf = current_history_fit.get(dg_id, {})
        win_hf = hf.get("win", 0.0) * 100

        # Collect historical finish text for each year
        year_results = []
        for year in HISTORY_YEARS:
            yr_data = history[year].get(dg_id)
            if yr_data:
                fin = yr_data.get("fin_text", "---")
                year_results.append(f"{fin:>6}")
            else:
                year_results.append(f"{'---':>6}")

        hist_str = "  ".join(year_results)
        print(f"{rank:<5} {name:<28} {win_base:>10.1f}%  {win_hf:>11.1f}%   {hist_str}")

    print()
    print("Columns: Win%(Base) = skill model | Win%(+Hist) = skill + course history/fit")
    print(f"History years: {', '.join(str(y) for y in HISTORY_YEARS)}  (--- = not in field that year)")
    print()

    # Summary: who has the best course history among current top players
    print("--- Players with wins or top-5 finishes at this course ---")
    found_any = False
    for p in field_players:
        dg_id = p["dg_id"]
        name = p["player_name"]
        notable = []
        for year in HISTORY_YEARS:
            yr_data = history[year].get(dg_id)
            if yr_data:
                fin = yr_data.get("fin_text", "")
                if finish_sort_key(fin) <= 5:
                    notable.append(f"{year}: {fin}")
        if notable:
            found_any = True
            print(f"  {name:<30} {', '.join(notable)}")

    if not found_any:
        print("  No players in current field have a win or top-5 finish at this course in available history.")


if __name__ == "__main__":
    main()
