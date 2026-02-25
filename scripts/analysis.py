#!/usr/bin/env python3
"""
Full-field analysis: DataGolf model vs Kalshi market vs course history.

Shows:
  - DataGolf win% (baseline skill model)
  - DataGolf win% with course history/fit adjustment
  - Course fit bump: how much the history model moves each player vs baseline
  - Kalshi market-implied win%
  - Model/market edge: positive = Kalshi higher (market likes them more than DG)
  - Past 3 years of finishes at this course
  - Flags: course specialist, model/market divergence
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pga_oad import DataGolfClient
from pga_oad.kalshi import KalshiClient
from pga_oad.blend import SignalBlender, _match_player
from pga_oad.models import Prediction

HISTORY_YEARS = [2025, 2024, 2023]
FULL_HISTORY_YEARS = [2025, 2024, 2023, 2022, 2021, 2020]


def finish_sort_key(fin: str) -> int:
    if not fin or fin == "---":
        return 9999
    clean = fin.lstrip("T").lstrip("=")
    try:
        return int(clean)
    except ValueError:
        return {"CUT": 200, "WD": 300, "DQ": 300, "MDF": 199}.get(fin, 9999)


def main():
    dg = DataGolfClient()
    kalshi = KalshiClient()

    # ── DataGolf ──────────────────────────────────────────────────
    print("Fetching DataGolf predictions...")
    dg_data = dg.get_pre_tournament_predictions(tour="pga", odds_format="percent")
    event_name = dg_data.get("event_name", "Current Event")
    baseline_raw = {p["dg_id"]: p for p in dg_data.get("baseline", [])}
    history_fit_raw = {p["dg_id"]: p for p in dg_data.get("baseline_history_fit", [])}

    predictions = []
    for p in dg_data.get("baseline", []):
        try:
            predictions.append(Prediction(
                player_name=p["player_name"],
                dg_id=p["dg_id"],
                win_prob=p.get("win", 0.0),
                top5_prob=p.get("top_5", 0.0),
                top10_prob=p.get("top_10", 0.0),
                top20_prob=p.get("top_20", 0.0),
                make_cut_prob=p.get("make_cut", 0.0),
            ))
        except Exception:
            continue

    # ── Kalshi ────────────────────────────────────────────────────
    print("Detecting Kalshi event code...")
    event_code = kalshi.detect_current_event_code()
    kalshi_win_probs: dict[str, float] = {}
    if event_code:
        print(f"Fetching Kalshi win markets for: {event_code}")
        kalshi_win_probs = kalshi.get_implied_probs(event_code, "win")
    else:
        print("No open Kalshi golf markets found.")

    # ── Course history (archive) ──────────────────────────────────
    # Detect event_id from schedule
    event_id = None
    print("Fetching schedule to find event ID...")
    sched = dg.get_schedule(tour="pga", upcoming_only=True)
    for e in sched.get("schedule", []):
        if event_name.lower() in e.get("event_name", "").lower() or \
           e.get("event_name", "").lower() in event_name.lower():
            event_id = e.get("event_id")
            break
    if event_id is None and sched.get("schedule"):
        event_id = sched["schedule"][0]["event_id"]

    history: dict[int, dict[int, dict]] = {}  # year → {dg_id → player_data}
    if event_id:
        print(f"Fetching course history (event_id={event_id}, years={FULL_HISTORY_YEARS})...")
        for year in FULL_HISTORY_YEARS:
            try:
                arch = dg.get_pre_tournament_archive(event_id=event_id, year=year)
                history[year] = {p["dg_id"]: p for p in arch.get("baseline", [])}
            except Exception:
                history[year] = {}

    # ── Build combined rows ───────────────────────────────────────
    rows = []
    for pred in predictions:
        dg_id = pred.dg_id
        name = pred.player_name

        base_win = baseline_raw.get(dg_id, {}).get("win", pred.win_prob)
        hf_win = history_fit_raw.get(dg_id, {}).get("win", base_win)
        fit_bump = hf_win - base_win  # positive = history helps this player

        kal_win = _match_player(name, kalshi_win_probs)
        edge = (kal_win - hf_win) if kal_win is not None else None  # positive = Kalshi higher

        # Course history: collect finishes and best finish
        year_fins: dict[int, str] = {}
        for year in FULL_HISTORY_YEARS:
            yr = history.get(year, {}).get(dg_id)
            year_fins[year] = yr.get("fin_text", "---") if yr else "---"

        best_fin = min(
            (finish_sort_key(f) for f in year_fins.values() if f != "---"),
            default=9999,
        )

        rows.append({
            "name": name,
            "dg_id": dg_id,
            "base_win": base_win,
            "hf_win": hf_win,
            "fit_bump": fit_bump,
            "top10": pred.top10_prob,
            "top20": pred.top20_prob,
            "make_cut": pred.make_cut_prob,
            "kal_win": kal_win,
            "edge": edge,
            "year_fins": year_fins,
            "best_fin": best_fin,
        })

    rows.sort(key=lambda r: r["hf_win"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["dg_rank"] = i

    if kalshi_win_probs:
        kal_sorted = sorted(
            [r for r in rows if r["kal_win"] is not None],
            key=lambda r: r["kal_win"],
            reverse=True,
        )
        for i, r in enumerate(kal_sorted, 1):
            r["kal_rank"] = i
    for r in rows:
        r.setdefault("kal_rank", None)

    # ── SECTION 1: Full field table ───────────────────────────────
    print(f"\n{'='*140}")
    print(f"  {event_name}  —  Full Field Analysis")
    print(f"{'='*140}")
    print(f"  DataGolf model (DG) vs Kalshi prediction market (Kal)  |  Course history: {', '.join(str(y) for y in HISTORY_YEARS)}")
    print(f"{'='*140}")

    hdr = (f"{'DGR':>4} {'KalR':>4}  {'Player':<28}  "
           f"{'DG Base%':>9} {'DG+Hist%':>9} {'Fit':>5}  "
           f"{'Kal%':>7} {'Edge':>6}  "
           f"{'Top10%':>7} {'Cut%':>5}  "
           + "  ".join(f"{y}" for y in HISTORY_YEARS))
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        kal_str = f"{r['kal_win']*100:>6.1f}%" if r["kal_win"] is not None else f"{'---':>7}"
        edge_str = f"{r['edge']*100:>+5.1f}%" if r["edge"] is not None else f"{'---':>6}"
        fit_str = f"{r['fit_bump']*100:>+4.1f}%"
        kal_rank_str = f"{r['kal_rank']:>4}" if r["kal_rank"] else f"{'---':>4}"
        hist_str = "  ".join(f"{r['year_fins'].get(y,'---'):>5}" for y in HISTORY_YEARS)

        # Flag course specialists (top 10 finish in last 3 years)
        specialist = " *" if r["best_fin"] <= 10 else "  "

        print(
            f"{r['dg_rank']:>4} {kal_rank_str}  {r['name']:<28}  "
            f"{r['base_win']*100:>8.1f}% {r['hf_win']*100:>8.1f}% {fit_str}  "
            f"{kal_str} {edge_str}  "
            f"{r['top10']*100:>6.1f}% {r['make_cut']*100:>4.1f}%  "
            f"{hist_str}{specialist}"
        )

    print(f"\n  * = top-10 finish at this course in last 3 years")
    print(f"  Fit = DG+Hist% minus DG Base% (positive = course history helps this player)")
    print(f"  Edge = Kal% minus DG+Hist% (positive = market more bullish than model)")

    # ── SECTION 2: Course specialists ────────────────────────────
    print(f"\n{'='*100}")
    print("  COURSE SPECIALISTS  (top-10 finish at this course in last 3 years)")
    print(f"{'='*100}")
    specialists = sorted([r for r in rows if r["best_fin"] <= 10], key=lambda r: r["best_fin"])
    if specialists:
        print(f"  {'Player':<28} {'DG Rank':>8} {'DG+Hist%':>9} {'Kal%':>7} {'Edge':>6}  History (2025/2024/2023/2022/2021/2020)")
        print(f"  {'-'*100}")
        for r in specialists:
            kal_str = f"{r['kal_win']*100:>6.1f}%" if r["kal_win"] is not None else f"{'---':>7}"
            edge_str = f"{r['edge']*100:>+5.1f}%" if r["edge"] is not None else "   ---"
            hist = " / ".join(r["year_fins"].get(y, "---") for y in FULL_HISTORY_YEARS)
            print(f"  {r['name']:<28} {r['dg_rank']:>8} {r['hf_win']*100:>8.1f}% {kal_str} {edge_str}  {hist}")
    else:
        print("  No players in current field with a top-10 at this course in 2023-2025.")

    # ── SECTION 3: Model vs market outliers ──────────────────────
    print(f"\n{'='*100}")
    print("  MODEL / MARKET OUTLIERS  (DataGolf and Kalshi disagree most)")
    print(f"{'='*100}")

    with_kal = [r for r in rows if r["edge"] is not None]

    # DataGolf much higher than Kalshi (model bullish, market skeptical)
    dg_higher = sorted(with_kal, key=lambda r: r["edge"])[:8]
    print(f"\n  >> DataGolf MORE bullish than Kalshi (model sees value the market doesn't)")
    print(f"  {'Player':<28} {'DG Rank':>8} {'DG+Hist%':>9} {'Kal%':>7} {'Edge':>6}  History")
    print(f"  {'-'*85}")
    for r in dg_higher:
        kal_str = f"{r['kal_win']*100:>6.1f}%"
        hist = " / ".join(r["year_fins"].get(y, "---") for y in HISTORY_YEARS)
        print(f"  {r['name']:<28} {r['dg_rank']:>8} {r['hf_win']*100:>8.1f}% {kal_str} {r['edge']*100:>+5.1f}%  {hist}")

    # Kalshi much higher than DataGolf (market bullish, model skeptical)
    kal_higher = sorted(with_kal, key=lambda r: r["edge"], reverse=True)[:8]
    print(f"\n  >> Kalshi MORE bullish than DataGolf (market sees value the model doesn't)")
    print(f"  {'Player':<28} {'DG Rank':>8} {'DG+Hist%':>9} {'Kal%':>7} {'Edge':>6}  History")
    print(f"  {'-'*85}")
    for r in kal_higher:
        kal_str = f"{r['kal_win']*100:>6.1f}%"
        hist = " / ".join(r["year_fins"].get(y, "---") for y in HISTORY_YEARS)
        print(f"  {r['name']:<28} {r['dg_rank']:>8} {r['hf_win']*100:>8.1f}% {kal_str} {r['edge']*100:>+5.1f}%  {hist}")

    # ── SECTION 4: Biggest course history movers ─────────────────
    print(f"\n{'='*100}")
    print("  COURSE FIT MOVERS  (biggest difference between DG baseline and DG+course history)")
    print(f"{'='*100}")

    boosted = sorted(rows, key=lambda r: r["fit_bump"], reverse=True)[:8]
    dropped = sorted(rows, key=lambda r: r["fit_bump"])[:8]

    print(f"\n  >> Most BOOSTED by course history (history model likes them more than baseline)")
    print(f"  {'Player':<28} {'DG Rank':>8} {'Base%':>6} {'Hist%':>7} {'Bump':>6}  History")
    print(f"  {'-'*80}")
    for r in boosted:
        hist = " / ".join(r["year_fins"].get(y, "---") for y in HISTORY_YEARS)
        print(f"  {r['name']:<28} {r['dg_rank']:>8} {r['base_win']*100:>5.1f}% {r['hf_win']*100:>6.1f}% {r['fit_bump']*100:>+5.1f}%  {hist}")

    print(f"\n  >> Most DROPPED by course history (history model likes them less than baseline)")
    print(f"  {'Player':<28} {'DG Rank':>8} {'Base%':>6} {'Hist%':>7} {'Bump':>6}  History")
    print(f"  {'-'*80}")
    for r in dropped:
        hist = " / ".join(r["year_fins"].get(y, "---") for y in HISTORY_YEARS)
        print(f"  {r['name']:<28} {r['dg_rank']:>8} {r['base_win']*100:>5.1f}% {r['hf_win']*100:>6.1f}% {r['fit_bump']*100:>+5.1f}%  {hist}")


if __name__ == "__main__":
    main()
