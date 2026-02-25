#!/usr/bin/env python3
"""Fetch this week's field and pre-tournament win probabilities from DataGolf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pga_oad import DataGolfClient
from pga_oad.models import Prediction


def main():
    client = DataGolfClient()
    data = client.get_pre_tournament_predictions(tour="pga", odds_format="percent")

    event_name = data.get("event_name", "Current Event")
    last_updated = data.get("last_updated", "")
    # DataGolf returns two models: baseline (skill only) and baseline_history_fit (skill + course history)
    baseline = data.get("baseline", [])
    history_fit = {p["dg_id"]: p for p in data.get("baseline_history_fit", [])}

    if not baseline:
        print("No prediction data available. Is there an active/upcoming tournament?")
        return

    # API returns probabilities as decimals (0.0-1.0), not percentages
    def parse_pred(p: dict) -> Prediction:
        return Prediction(
            player_name=p.get("player_name", ""),
            dg_id=p.get("dg_id", 0),
            win_prob=p.get("win", 0.0),
            top5_prob=p.get("top_5", 0.0),
            top10_prob=p.get("top_10", 0.0),
            top20_prob=p.get("top_20", 0.0),
            make_cut_prob=p.get("make_cut", 0.0),
        )

    preds = []
    for p in baseline:
        try:
            preds.append(parse_pred(p))
        except Exception:
            continue

    preds.sort(key=lambda p: p.win_prob, reverse=True)

    models_available = data.get("models_available", [])
    has_history_fit = "baseline_history_fit" in models_available

    print(f"\nEvent: {event_name}")
    print(f"Last updated: {last_updated}")

    if has_history_fit:
        print(f"\n{'Rank':<6} {'Player':<30} {'Win%(Base)':>11} {'Win%(+Hist)':>12} {'Top10%':>8} {'Top20%':>8} {'Cut%':>6} {'EV(Base)':>9}")
        print("-" * 110)
        for rank, pred in enumerate(preds, 1):
            hf = history_fit.get(pred.dg_id, {})
            win_hf = hf.get("win", 0.0)
            print(
                f"{rank:<6} {pred.player_name:<30}"
                f" {pred.win_prob*100:>10.1f}%"
                f" {win_hf*100:>11.1f}%"
                f" {pred.top10_prob*100:>7.1f}%"
                f" {pred.top20_prob*100:>7.1f}%"
                f" {pred.make_cut_prob*100:>5.1f}%"
                f" {pred.expected_value:>9.4f}"
            )
    else:
        print(f"\n{'Rank':<6} {'Player':<30} {'Win%':>6} {'Top5%':>7} {'Top10%':>8} {'Top20%':>8} {'Cut%':>6} {'EV':>7}")
        print("-" * 90)
        for rank, pred in enumerate(preds, 1):
            print(
                f"{rank:<6} {pred.player_name:<30}"
                f" {pred.win_prob*100:>5.1f}%"
                f" {pred.top5_prob*100:>6.1f}%"
                f" {pred.top10_prob*100:>7.1f}%"
                f" {pred.top20_prob*100:>7.1f}%"
                f" {pred.make_cut_prob*100:>5.1f}%"
                f" {pred.expected_value:>7.4f}"
            )

    print(f"\nField size: {len(preds)} players")
    if has_history_fit:
        print("Win%(Base) = skill model only | Win%(+Hist) = skill + course history & fit")


if __name__ == "__main__":
    main()
