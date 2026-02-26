"""
Backtesting engine: simulates model picks against historical results.

This is the #1 feature that sells subscriptions. When users can see
"this model picked 3 winners and had a 14.2 avg finish over 40 events",
they trust it enough to pay.

Strategy:
  For each completed tournament in the season:
    1. Fetch the pre-tournament archive (what predictions looked like before the event)
    2. Pick the top-ranked available player (simulating one-and-done constraint)
    3. Fetch actual results and score the pick

  Scoring:
    Win = 30 pts, Top-5 = 15 pts, Top-10 = 8 pts, Top-20 = 4 pts,
    Made cut = 1 pt, Missed cut = 0 pts
"""
from __future__ import annotations

import logging
from typing import Optional

from ..client import DataGolfClient

logger = logging.getLogger(__name__)

# Points system for backtesting
POINTS = {
    "win": 30.0,
    "top5": 15.0,
    "top10": 8.0,
    "top20": 4.0,
    "made_cut": 1.0,
    "missed_cut": 0.0,
}


class BacktestEngine:
    """Simulate model performance over a historical season."""

    def __init__(self, client: DataGolfClient | None = None):
        self.client = client or DataGolfClient()

    def run(
        self,
        season: int = 2024,
        model_name: str = "proprietary",
        exclude_dg_ids: set[int] | None = None,
    ) -> dict:
        """
        Run a full-season backtest.

        For each completed event in the season:
          1. Get pre-tournament predictions (archive)
          2. Select the top available player (one-and-done: no repeats)
          3. Look up actual result from historical data
          4. Score the pick

        Args:
            season: PGA Tour season year to backtest
            model_name: Which model signal to use for ranking
            exclude_dg_ids: Players to exclude (pre-used)

        Returns:
            Dict with aggregate stats and per-event details
        """
        used_ids: set[int] = set(exclude_dg_ids or [])
        event_details: list[dict] = []

        # Fetch schedule for the season
        try:
            schedule_data = self.client.get_schedule(tour="pga", season=season)
            events = schedule_data.get("schedule", [])
        except Exception as e:
            logger.error("Failed to fetch schedule for season %d: %s", season, e)
            return self._empty_result(season, model_name)

        completed = [
            e for e in events
            if e.get("status") == "completed"
        ]
        completed.sort(key=lambda e: e.get("start_date", ""))

        for event in completed:
            event_id = int(event["event_id"])
            event_name = event.get("event_name", f"Event {event_id}")

            # Get pre-tournament predictions for this event
            try:
                archive = self.client.get_pre_tournament_archive(
                    event_id=event_id, year=season, odds_format="percent"
                )
            except Exception:
                # Try adjacent years for PGA fiscal calendar
                archive = None
                for fallback_year in [season + 1, season - 1]:
                    try:
                        archive = self.client.get_pre_tournament_archive(
                            event_id=event_id, year=fallback_year, odds_format="percent"
                        )
                        if archive and archive.get("baseline"):
                            break
                    except Exception:
                        continue
                if not archive:
                    continue

            baseline = archive.get("baseline", [])
            history_fit = archive.get("baseline_history_fit", [])

            if not baseline:
                continue

            # Use history_fit if available, else baseline
            ranking_source = history_fit if history_fit else baseline

            # Sort by win probability and pick the top available player
            ranked = sorted(
                ranking_source,
                key=lambda p: float(p.get("win", 0)),
                reverse=True,
            )

            picked = None
            for player in ranked:
                dg_id = player["dg_id"]
                if dg_id not in used_ids:
                    picked = player
                    used_ids.add(dg_id)
                    break

            if picked is None:
                continue

            # Look up actual result
            fin_text = picked.get("fin_text", "")
            actual_finish = self._parse_finish(fin_text)
            points = self._score_finish(actual_finish, fin_text)

            event_details.append({
                "event_id": event_id,
                "event_name": event_name,
                "player_name": picked["player_name"],
                "dg_id": picked["dg_id"],
                "pre_tournament_win_prob": round(float(picked.get("win", 0)) * 100, 2),
                "finish_text": fin_text,
                "finish_position": actual_finish,
                "points_earned": points,
            })

        return self._aggregate(season, model_name, event_details)

    def _parse_finish(self, fin_text: str) -> int | None:
        """Parse finish text to numeric position."""
        if not fin_text or fin_text in ("CUT", "WD", "DQ", "MDF", "DNF", "-", "---"):
            return None
        clean = fin_text.lstrip("T").lstrip("=")
        try:
            return int(clean)
        except ValueError:
            return None

    def _score_finish(self, position: int | None, fin_text: str) -> float:
        """Score a finish position using the points system."""
        if position is None:
            if fin_text in ("CUT", "MDF"):
                return POINTS["missed_cut"]
            return POINTS["missed_cut"]

        if position == 1:
            return POINTS["win"]
        if position <= 5:
            return POINTS["top5"]
        if position <= 10:
            return POINTS["top10"]
        if position <= 20:
            return POINTS["top20"]
        return POINTS["made_cut"]

    def _aggregate(self, season: int, model_name: str, details: list[dict]) -> dict:
        """Aggregate per-event results into summary stats."""
        if not details:
            return self._empty_result(season, model_name)

        total_points = sum(d["points_earned"] for d in details)
        finishes = [d["finish_position"] for d in details if d["finish_position"] is not None]
        avg_finish = sum(finishes) / len(finishes) if finishes else None

        return {
            "season": season,
            "model_version": model_name,
            "total_events": len(details),
            "total_points": round(total_points, 2),
            "avg_finish": round(avg_finish, 1) if avg_finish else None,
            "win_count": sum(1 for d in details if d["finish_position"] == 1),
            "top5_count": sum(1 for d in details if d["finish_position"] and d["finish_position"] <= 5),
            "top10_count": sum(1 for d in details if d["finish_position"] and d["finish_position"] <= 10),
            "top20_count": sum(1 for d in details if d["finish_position"] and d["finish_position"] <= 20),
            "cut_count": sum(1 for d in details if d["finish_position"] is None),
            "roi_estimate": round(total_points / len(details), 2) if details else None,
            "event_details": details,
        }

    def _empty_result(self, season: int, model_name: str) -> dict:
        return {
            "season": season,
            "model_version": model_name,
            "total_events": 0,
            "total_points": 0.0,
            "avg_finish": None,
            "win_count": 0,
            "top5_count": 0,
            "top10_count": 0,
            "top20_count": 0,
            "cut_count": 0,
            "roi_estimate": None,
            "event_details": [],
        }
