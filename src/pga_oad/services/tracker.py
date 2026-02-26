"""
Pick performance tracker: resolves picks against actual tournament results.

Runs after each tournament completes to:
  1. Fetch actual results from DataGolf historical data
  2. Match each user's pick to actual finish
  3. Score and store the result
  4. Update season performance metrics
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from ..client import DataGolfClient
from ..db import crud
from ..db.models import Pick, PickResult

logger = logging.getLogger(__name__)

# Same scoring as backtester for consistency
POINTS = {
    "win": 30.0,
    "top5": 15.0,
    "top10": 8.0,
    "top20": 4.0,
    "made_cut": 1.0,
    "missed_cut": 0.0,
}


class PerformanceTracker:
    """Resolves picks against actual tournament results."""

    def __init__(self, client: DataGolfClient | None = None):
        self.client = client or DataGolfClient()

    def resolve_event(self, db: Session, event_id: int, season: int) -> list[dict]:
        """
        Resolve all picks for a completed tournament.

        Fetches the actual results from DataGolf, matches each pick
        to the player's actual finish, and stores PickResult records.

        Returns list of resolution summaries.
        """
        # Fetch actual results
        try:
            archive = self.client.get_pre_tournament_archive(
                event_id=event_id, year=season, odds_format="percent"
            )
        except Exception as e:
            logger.error("Failed to fetch results for event %d: %s", event_id, e)
            return []

        # Build lookup: dg_id -> result
        baseline = archive.get("baseline", [])
        results_map: dict[int, dict] = {}
        for player in baseline:
            results_map[player["dg_id"]] = player

        # Find all unresolved picks for this event
        picks = (
            db.query(Pick)
            .filter(Pick.event_id == event_id, Pick.season == season)
            .all()
        )

        resolutions = []
        for pick in picks:
            # Skip already resolved
            if pick.result is not None:
                continue

            player_result = results_map.get(pick.dg_id)
            if player_result is None:
                continue

            fin_text = player_result.get("fin_text", "")
            position = self._parse_finish(fin_text)
            made_cut = position is not None and fin_text not in ("CUT", "MDF", "WD", "DQ")
            points = self._score_finish(position, fin_text)

            # Compute percentile vs field
            total_field = len(baseline)
            if position and total_field > 0:
                score_vs_field = round((1.0 - position / total_field) * 100, 1)
            else:
                score_vs_field = 0.0

            crud.record_pick_result(
                db,
                pick_id=pick.id,
                finish_position=position,
                finish_text=fin_text,
                made_cut=made_cut,
                score_vs_field=score_vs_field,
                points_earned=points,
            )

            resolutions.append({
                "pick_id": pick.id,
                "player_name": pick.player_name,
                "finish_text": fin_text,
                "finish_position": position,
                "points_earned": points,
                "score_vs_field": score_vs_field,
            })

        return resolutions

    def _parse_finish(self, fin_text: str) -> int | None:
        if not fin_text or fin_text in ("CUT", "WD", "DQ", "MDF", "DNF", "-", "---"):
            return None
        clean = fin_text.lstrip("T").lstrip("=")
        try:
            return int(clean)
        except ValueError:
            return None

    def _score_finish(self, position: int | None, fin_text: str) -> float:
        if position is None:
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
