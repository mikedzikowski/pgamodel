#!/usr/bin/env python3
"""
Blended ranking: DataGolf model + Kalshi prediction market signals.

Shows side-by-side: DG rank, Kalshi rank, blended rank, and edge
(positive edge = Kalshi market implies higher win% than DG model).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pga_oad import DataGolfClient
from pga_oad.kalshi import KalshiClient
from pga_oad.blend import SignalBlender
from pga_oad.models import Prediction


def parse_probs(data: dict) -> tuple[list[Prediction], dict[int, dict]]:
    """Parse DataGolf pre-tournament response into Prediction list + history_fit map."""
    baseline = data.get("baseline", [])
    history_fit_map = {p["dg_id"]: p for p in data.get("baseline_history_fit", [])}

    predictions = []
    for p in baseline:
        try:
            predictions.append(
                Prediction(
                    player_name=p.get("player_name", ""),
                    dg_id=p.get("dg_id", 0),
                    win_prob=p.get("win", 0.0),
                    top5_prob=p.get("top_5", 0.0),
                    top10_prob=p.get("top_10", 0.0),
                    top20_prob=p.get("top_20", 0.0),
                    make_cut_prob=p.get("make_cut", 0.0),
                )
            )
        except Exception:
            continue

    return predictions, history_fit_map


def main():
    parser = argparse.ArgumentParser(description="Blended DataGolf + Kalshi rankings")
    parser.add_argument("--top", type=int, default=40, help="Show top N players (default: 40)")
    parser.add_argument(
        "--w-dg", type=float, default=0.65,
        help="Weight for DataGolf model (default: 0.65)",
    )
    parser.add_argument(
        "--w-kalshi", type=float, default=0.35,
        help="Weight for Kalshi market (default: 0.35)",
    )
    parser.add_argument(
        "--edge-only", action="store_true",
        help="Show only players where Kalshi diverges meaningfully from DataGolf (|edge| > 1%%)",
    )
    parser.add_argument(
        "--matchups", action="store_true",
        help="Also show available head-to-head matchup markets",
    )
    args = parser.parse_args()

    if abs(args.w_dg + args.w_kalshi - 1.0) > 0.01:
        print("Error: --w-dg and --w-kalshi must sum to 1.0")
        return

    dg_client = DataGolfClient()
    kalshi_client = KalshiClient()
    blender = SignalBlender(w_dg=args.w_dg, w_kalshi=args.w_kalshi)

    # --- DataGolf ---
    print("Fetching DataGolf predictions...")
    dg_data = dg_client.get_pre_tournament_predictions(tour="pga", odds_format="percent")
    event_name = dg_data.get("event_name", "Current Event")
    predictions, history_fit_map = parse_probs(dg_data)

    if not predictions:
        print("No DataGolf predictions available.")
        return

    # --- Kalshi ---
    print("Detecting Kalshi event code...")
    event_code = kalshi_client.detect_current_event_code()
    if not event_code:
        print("No open Kalshi golf markets found. Showing DataGolf only.")
        event_code = None

    kalshi_win_probs: dict[str, float] = {}
    kalshi_top10_probs: dict[str, float] = {}
    kalshi_top20_probs: dict[str, float] = {}
    kalshi_make_cut_probs: dict[str, float] = {}

    if event_code:
        print(f"Fetching Kalshi markets for event: {event_code}")
        kalshi_win_probs = kalshi_client.get_implied_probs(event_code, "win")
        kalshi_top10_probs = kalshi_client.get_implied_probs(event_code, "top10", normalize=False)
        kalshi_top20_probs = kalshi_client.get_implied_probs(event_code, "top20", normalize=False)
        kalshi_make_cut_probs = kalshi_client.get_implied_probs(event_code, "make_cut", normalize=False)
        print(f"  Win markets: {len(kalshi_win_probs)} players")
        print(f"  Top10 markets: {len(kalshi_top10_probs)} players")
        print(f"  Make cut markets: {len(kalshi_make_cut_probs)} players")

    # --- Blend ---
    blended = blender.blend(
        predictions,
        history_fit_map,
        kalshi_win_probs,
        kalshi_top10_probs=kalshi_top10_probs,
        kalshi_top20_probs=kalshi_top20_probs,
        kalshi_make_cut_probs=kalshi_make_cut_probs,
    )

    # --- Filter ---
    if args.edge_only:
        display = [p for p in blended if abs(p.edge) > 0.01]
        display.sort(key=lambda p: abs(p.edge), reverse=True)
    else:
        display = blended[:args.top]

    # --- Print ---
    kalshi_coverage = sum(1 for p in blended if p.kalshi_win_prob is not None)
    print(f"\nEvent: {event_name}")
    print(f"Kalshi coverage: {kalshi_coverage}/{len(blended)} players  |  "
          f"Weights: DG={args.w_dg:.0%}  Kalshi={args.w_kalshi:.0%}")

    if event_code:
        print(f"\n{'Rank':<5} {'Blnd':<5} {'DG':<5} {'Kal':<5} "
              f"{'Player':<28} {'DG Win%':>8} {'Kal Win%':>9} {'Blnd Win%':>10} "
              f"{'Edge':>6} {'Top10%':>7} {'Cut%':>6}")
        print("-" * 120)

        for p in display:
            kal_str = f"{p.kalshi_win_prob*100:>8.1f}%" if p.kalshi_win_prob is not None else f"{'---':>9}"
            kal_rank_str = f"{p.kalshi_rank}" if p.kalshi_rank > 0 else "---"
            edge_str = f"{p.edge*100:>+5.1f}%" if p.kalshi_win_prob is not None else f"{'---':>6}"
            # Top10: prefer Kalshi if available, else DataGolf
            top10 = p.kalshi_top10_prob if p.kalshi_top10_prob is not None else p.dg_top10_prob
            cut = p.kalshi_make_cut_prob if p.kalshi_make_cut_prob is not None else p.dg_make_cut_prob
            print(
                f"{p.blended_rank:<5} {p.blended_rank:<5} {p.dg_rank:<5} {kal_rank_str:<5}"
                f" {p.player_name:<28}"
                f" {p.dg_win_prob_history_fit*100:>7.1f}%"
                f" {kal_str}"
                f" {p.blended_win_prob*100:>9.1f}%"
                f" {edge_str}"
                f" {top10*100:>6.1f}%"
                f" {cut*100:>5.1f}%"
            )
    else:
        # DataGolf only fallback
        print(f"\n{'Rank':<5} {'Player':<30} {'DG Win%(+Hist)':>15} {'Top10%':>8} {'Cut%':>6}")
        print("-" * 75)
        for i, p in enumerate(display, 1):
            print(f"{i:<5} {p.player_name:<30} {p.dg_win_prob_history_fit*100:>14.1f}%"
                  f" {p.dg_top10_prob*100:>7.1f}% {p.dg_make_cut_prob*100:>5.1f}%")

    # Edge leaders summary
    if event_code and not args.edge_only:
        print("\n--- Biggest Kalshi vs DataGolf divergences (market disagreements) ---")
        edge_leaders = sorted(
            [p for p in blended if p.kalshi_win_prob is not None],
            key=lambda p: abs(p.edge),
            reverse=True,
        )[:8]
        for p in edge_leaders:
            direction = "Kalshi HIGHER" if p.edge > 0 else "DataGolf HIGHER"
            print(
                f"  {p.player_name:<28} DG={p.dg_win_prob_history_fit*100:.1f}%  "
                f"Kal={p.kalshi_win_prob*100:.1f}%  "
                f"Edge={p.edge*100:+.1f}%  ({direction})"
            )

    # Matchups
    if args.matchups and event_code:
        matchups = kalshi_client.get_matchups(event_code)
        if matchups:
            print(f"\n--- Head-to-Head Matchups ({len(matchups)} markets) ---")
            for m in sorted(matchups, key=lambda x: abs(x["implied_prob"] - 0.5)):
                print(f"  {m['title'][:70]:<70}  {m['implied_prob']*100:.0f}%")


if __name__ == "__main__":
    main()
