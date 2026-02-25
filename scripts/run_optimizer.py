#!/usr/bin/env python3
"""
Run the One and Done season-long optimizer.

Fetches the upcoming PGA Tour schedule, pulls pre-tournament predictions
for each event, and outputs an optimal pick assignment for the season.
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pga_oad import DataGolfClient, OneAndDoneOptimizer
from pga_oad.models import Prediction, Tournament


def parse_used_players(value: str) -> set[int]:
    """Parse comma-separated dg_ids of already-used players."""
    if not value:
        return set()
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def main():
    parser = argparse.ArgumentParser(description="One and Done season-long optimizer")
    parser.add_argument(
        "--used",
        default="",
        help="Comma-separated dg_ids of players already used this season",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (default: current)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save picks to data/processed/picks_YYYY-MM-DD.csv",
    )
    args = parser.parse_args()

    used_players = parse_used_players(args.used)
    if used_players:
        print(f"Excluding {len(used_players)} already-used player(s).")

    client = DataGolfClient()

    print("Fetching schedule...")
    schedule_data = client.get_schedule(tour="pga", season=args.season, upcoming_only=True)
    events = schedule_data.get("schedule", [])

    if not events:
        print("No upcoming events found.")
        return

    print(f"Found {len(events)} upcoming tournaments.")

    # Build Tournament objects
    tournaments: list[Tournament] = []
    for e in events:
        tournaments.append(
            Tournament(
                event_id=e.get("event_id", 0),
                event_name=e.get("event_name", ""),
                course=e.get("course", ""),
                date=e.get("date", ""),
                tour="pga",
            )
        )

    # Fetch pre-tournament predictions for the current/next event
    # (DataGolf only provides predictions for the active upcoming event)
    print("Fetching pre-tournament predictions for current event...")
    pred_data = client.get_pre_tournament_predictions(tour="pga", odds_format="percent")

    current_event_name = pred_data.get("event_name", "")
    raw_players = pred_data.get("baseline", [])

    if not raw_players:
        print("No prediction data available from DataGolf right now.")
        print("This typically means there is no active tournament week.")
        return

    # Match predictions to the current tournament
    current_tournament = None
    for t in tournaments:
        if t.event_name.lower() in current_event_name.lower() or current_event_name.lower() in t.event_name.lower():
            current_tournament = t
            break

    # Fallback: use the first upcoming tournament
    if current_tournament is None and tournaments:
        current_tournament = tournaments[0]

    # API returns probabilities as decimals (0.0-1.0), not percentages
    preds_for_event: list[Prediction] = []
    for p in raw_players:
        try:
            pred = Prediction(
                player_name=p.get("player_name", ""),
                dg_id=p.get("dg_id", 0),
                win_prob=p.get("win", 0.0),
                top5_prob=p.get("top_5", 0.0),
                top10_prob=p.get("top_10", 0.0),
                top20_prob=p.get("top_20", 0.0),
                make_cut_prob=p.get("make_cut", 0.0),
            )
            preds_for_event.append(pred)
        except Exception:
            continue

    predictions_map: dict[int, list[Prediction]] = {
        current_tournament.event_id: preds_for_event
    }
    active_tournaments = [current_tournament]

    print(f"\nRunning optimizer for: {current_tournament.event_name}")
    print(f"Field size: {len(preds_for_event)} players")

    optimizer = OneAndDoneOptimizer(used_players=used_players)

    try:
        picks = optimizer.optimize(active_tournaments, predictions_map)
    except Exception as e:
        print(f"Optimizer error: {e}")
        return

    # Print results
    print(f"\n{'Tournament':<45} {'Pick':<30} {'Win%':>6} {'EV':>7}")
    print("-" * 95)
    for pick in picks:
        print(
            f"{pick.event_name:<45} {pick.player_name:<30}"
            f" {pick.win_prob*100:>5.1f}%"
            f" {pick.expected_value:>7.4f}"
        )

    # Also show top 5 alternatives for this week
    print("\n--- Top 10 alternatives for this week (by EV) ---")
    sorted_preds = sorted(preds_for_event, key=lambda p: p.expected_value, reverse=True)
    print(f"\n{'Rank':<6} {'Player':<30} {'Win%':>6} {'EV':>7}")
    print("-" * 55)
    for i, p in enumerate(sorted_preds[:10], 1):
        marker = " <-- OPTIMAL" if picks and p.dg_id == picks[0].dg_id else ""
        print(f"{i:<6} {p.player_name:<30} {p.win_prob*100:>5.1f}% {p.expected_value:>7.4f}{marker}")

    if args.save:
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"picks_{date.today()}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["event_name", "player_name", "dg_id", "win_prob", "expected_value"])
            writer.writeheader()
            for pick in picks:
                writer.writerow(pick.model_dump())
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
