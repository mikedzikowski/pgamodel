"""
Proprietary PGA Win Prediction Model

Multi-signal ensemble that differs from DataGolf by combining:
  1. DataGolf skill + course history model (45% base weight)
  2. Sportsbook market consensus — vig-removed average across all books (30%)
  3. Kalshi prediction markets — liquid markets only (15%)
  4. Recency-weighted course history score — our own scoring (10%)

Weights are adaptive: when a signal is unavailable for a player the weight
is redistributed proportionally to the remaining signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .client import DataGolfClient
from .kalshi import KalshiClient

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

HISTORY_YEARS = [2025, 2024, 2023, 2022, 2021]

YEAR_WEIGHTS: dict[int, int] = {
    2025: 5,
    2024: 4,
    2023: 3,
    2022: 2,
    2021: 1,
}

# Base signal weights (must sum to 1.0)
BASE_WEIGHTS: dict[str, float] = {
    "dg":      0.45,
    "market":  0.30,
    "kalshi":  0.15,
    "history": 0.10,
}

# Fields in the outright odds response that are NOT sportsbook implied probs
# "datagolf" is a nested dict (baseline/history model), not a book
_NON_BOOK_FIELDS = {"player_name", "dg_id", "consensus_prob", "datagolf"}


# ──────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────

@dataclass
class ProprietaryPlayer:
    """All signals and outputs for one player in the proprietary model."""

    # Identity
    player_name: str          # DataGolf "Last, First" format
    dg_id: int

    # Input signals
    dg_win_prob: float                        # DG baseline model
    dg_win_prob_history: float                # DG skill + course history model
    market_consensus_prob: Optional[float]    # Vig-free sportsbook consensus
    dk_raw_prob: Optional[float]              # Raw DraftKings implied prob (with vig, for display)
    kalshi_win_prob: Optional[float]          # Kalshi implied probability
    recency_course_score: Optional[float]     # 0.0–1.0 recency-weighted history

    # Output
    proprietary_win_prob: float = 0.0
    proprietary_rank: int = 0
    dg_rank: int = 0
    rank_delta: int = 0   # dg_rank - proprietary_rank; positive = we rank higher
    weights_used: dict[str, float] = field(default_factory=dict)

    # Display
    finish_history: dict[int, str] = field(default_factory=dict)  # {year: fin_text}


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class ProprietaryModel:
    """
    Multi-signal ensemble model for PGA Tour win prediction.

    Usage:
        model = ProprietaryModel(client, kalshi_client)
        players = model.compute(event_id=10, event_code="COCITPB26")
    """

    def __init__(self, client: DataGolfClient, kalshi: KalshiClient) -> None:
        self.client = client
        self.kalshi = kalshi

    # ── Public ──────────────────────────────────────────────────────────────

    def compute(self, event_id: int, event_code: str) -> list[ProprietaryPlayer]:
        """
        Compute the proprietary model for the current tournament.

        Args:
            event_id:   DataGolf integer event ID (from schedule)
            event_code: Kalshi event code string (e.g. 'COCITPB26')

        Returns:
            List of ProprietaryPlayer sorted by proprietary_win_prob descending,
            with ranks assigned.
        """
        # ── Fetch all signals ──────────────────────────────────────────────
        dg_data = self.client.get_pre_tournament_predictions(tour="pga", odds_format="percent")
        odds_data = self.client.get_outright_odds(tour="pga", market="win", odds_format="percent")
        kalshi_probs = self._fetch_kalshi(event_code)
        history_map = self._fetch_course_history(event_id)

        # ── Parse DataGolf ─────────────────────────────────────────────────
        baseline_map: dict[int, dict] = {p["dg_id"]: p for p in dg_data.get("baseline", [])}
        history_fit_map: dict[int, dict] = {p["dg_id"]: p for p in dg_data.get("baseline_history_fit", [])}

        # ── Compute signals ────────────────────────────────────────────────
        market_probs = self._compute_market_consensus(odds_data)
        dk_raw_probs = self._extract_dk_raw(odds_data)
        raw_course_scores, finish_history_map = self._compute_course_scores(history_map)
        course_score_map = self._normalize_course_scores(raw_course_scores)

        # ── Build per-player output ────────────────────────────────────────
        players: list[ProprietaryPlayer] = []

        for dg_id, base in baseline_map.items():
            player_name = base["player_name"]
            dg_base_win = float(base.get("win", 0.0))
            dg_hist_win = float(history_fit_map.get(dg_id, {}).get("win", dg_base_win))

            # Outright odds API uses the same "Last, First" format as predictions — direct lookup
            market_prob = market_probs.get(player_name)
            dk_raw = dk_raw_probs.get(player_name)
            # Kalshi uses "First Last" format — needs name conversion
            kalshi_prob = self._match_player_name(player_name, kalshi_probs)
            course_score = course_score_map.get(dg_id)
            finish_hist = finish_history_map.get(dg_id, {})

            weights = self._compute_weights(
                kalshi_ok=kalshi_prob is not None,
                market_ok=market_prob is not None,
                history_ok=course_score is not None,
            )

            prop_prob = self._blend(
                dg_hist=dg_hist_win,
                market=market_prob,
                kalshi=kalshi_prob,
                course=course_score,
                weights=weights,
            )

            players.append(ProprietaryPlayer(
                player_name=player_name,
                dg_id=dg_id,
                dg_win_prob=dg_base_win,
                dg_win_prob_history=dg_hist_win,
                market_consensus_prob=market_prob,
                dk_raw_prob=dk_raw,
                kalshi_win_prob=kalshi_prob,
                recency_course_score=course_score,
                proprietary_win_prob=prop_prob,
                weights_used=weights,
                finish_history=finish_hist,
            ))

        return self._assign_ranks(players)

    # ── Private: data fetching ───────────────────────────────────────────────

    def _fetch_kalshi(self, event_code: str) -> dict[str, float]:
        """Fetch Kalshi win market implied probs (thin markets already excluded)."""
        if not event_code:
            return {}
        try:
            return self.kalshi.get_implied_probs(event_code, "win", normalize=True)
        except Exception:
            return {}

    def _fetch_course_history(self, event_id: int) -> dict[int, dict[int, dict]]:
        """
        Fetch archive data for HISTORY_YEARS.

        Returns:
            {year: {dg_id: player_archive_dict}}
        """
        result: dict[int, dict[int, dict]] = {}
        for year in HISTORY_YEARS:
            try:
                arch = self.client.get_pre_tournament_archive(
                    event_id=event_id,
                    year=year,
                    odds_format="percent",
                )
                result[year] = {p["dg_id"]: p for p in arch.get("baseline", [])}
            except Exception:
                result[year] = {}
        return result

    # ── Private: signal computation ─────────────────────────────────────────

    def _compute_market_consensus(self, odds_data: dict) -> dict[str, float]:
        """
        Extract vig-free implied win probabilities from sportsbook data.

        Strategy:
          1. Use DraftKings as the primary market signal (most liquid US book).
          2. If DraftKings is missing for a player, fall back to the average
             across all available books for that player.
          3. Normalize the full field to remove vig (overround), so probs sum to 1.0.

        The DataGolf API returns book odds as plain floats (implied probabilities
        with vig, e.g. 0.0526). The "datagolf" key is a nested dict and is excluded.

        Returns:
            {player_name: vig_free_prob}  names in DataGolf "Last, First" format
        """
        odds_list = odds_data.get("odds", [])
        if not odds_list:
            return {}

        raw_probs: dict[str, float] = {}

        for entry in odds_list:
            player_name = entry.get("player_name", "")
            if not player_name:
                continue

            # Prefer DraftKings; fall back to average across all books
            dk = entry.get("draftkings")
            if dk is not None and isinstance(dk, (int, float)) and float(dk) > 0:
                raw_probs[player_name] = float(dk)
            else:
                book_probs = [
                    float(v)
                    for k, v in entry.items()
                    if k not in _NON_BOOK_FIELDS
                    and isinstance(v, (int, float))
                    and float(v) > 0
                ]
                if book_probs:
                    raw_probs[player_name] = sum(book_probs) / len(book_probs)

        if not raw_probs:
            return {}

        # Vig removal: normalize so the full field sums to 1.0
        total = sum(raw_probs.values())
        if total <= 0:
            return {}

        return {name: prob / total for name, prob in raw_probs.items()}

    @staticmethod
    def _extract_dk_raw(odds_data: dict) -> dict[str, float]:
        """
        Extract raw DraftKings implied probabilities (before vig removal) for display.

        Returns:
            {player_name: raw_dk_implied_prob}  — includes vig, use for American odds display only
        """
        result: dict[str, float] = {}
        for entry in odds_data.get("odds", []):
            name = entry.get("player_name", "")
            dk = entry.get("draftkings")
            if name and dk is not None and isinstance(dk, (int, float)) and float(dk) > 0:
                result[name] = float(dk)
        return result

    def _compute_course_scores(
        self,
        history_map: dict[int, dict[int, dict]],
    ) -> tuple[dict[int, float], dict[int, dict[int, str]]]:
        """
        Compute recency-weighted course history scores.

        Scoring per year a player appeared:
          pos_score:  win=100, T2-5=65, T6-10=35, T11-20=15, T21-30=5, else=0
          contribution: pos_score * year_weight
          max_possible: 100 * year_weight

        raw_score = sum(contributions) / sum(max_possibles)  →  0.0–1.0

        Players with "-" or missing fin_text are skipped (archive artifact).

        Returns:
            (raw_scores, finish_history)
            raw_scores:     {dg_id: 0.0–1.0}  (normalized per participation)
            finish_history: {dg_id: {year: fin_text}}
        """
        raw_scores: dict[int, float] = {}
        finish_history: dict[int, dict[int, str]] = {}

        all_player_ids: set[int] = set()
        for year_data in history_map.values():
            all_player_ids.update(year_data.keys())

        for dg_id in all_player_ids:
            weighted_sum = 0.0
            max_possible = 0.0
            year_fins: dict[int, str] = {}

            for year in HISTORY_YEARS:
                player_entry = history_map.get(year, {}).get(dg_id)
                if player_entry is None:
                    continue

                fin_text = player_entry.get("fin_text", "")

                # Skip archive artifacts (player listed but no real finish)
                if not fin_text or fin_text in ("-", "---", "—"):
                    year_fins[year] = "—"
                    continue

                year_fins[year] = fin_text
                weight = YEAR_WEIGHTS[year]
                weighted_sum += self._position_score(fin_text) * weight
                max_possible += 100.0 * weight

            finish_history[dg_id] = year_fins

            if max_possible > 0:
                raw_scores[dg_id] = weighted_sum / max_possible

        return raw_scores, finish_history

    @staticmethod
    def _position_score(fin_text: str) -> float:
        """Convert fin_text to a 0–100 position score."""
        if not fin_text or fin_text in ("CUT", "WD", "DQ", "MDF", "DNF"):
            return 0.0
        clean = fin_text.lstrip("T").lstrip("=")
        try:
            pos = int(clean)
        except ValueError:
            return 0.0
        if pos == 1:
            return 100.0
        if pos <= 5:
            return 65.0
        if pos <= 10:
            return 35.0
        if pos <= 20:
            return 15.0
        if pos <= 30:
            return 5.0
        return 0.0

    @staticmethod
    def _normalize_course_scores(raw_scores: dict[int, float]) -> dict[int, float]:
        """
        Normalize recency scores to 0–1 relative to the best performer in the field.

        Players with no history are excluded (not in raw_scores → get None from .get()).
        Players with score 0.0 (all CUTs) remain at 0.0.
        """
        if not raw_scores:
            return {}
        max_score = max(raw_scores.values())
        if max_score <= 0:
            return {dg_id: 0.0 for dg_id in raw_scores}
        return {dg_id: score / max_score for dg_id, score in raw_scores.items()}

    # ── Private: weighting and blending ─────────────────────────────────────

    @staticmethod
    def _compute_weights(
        kalshi_ok: bool,
        market_ok: bool,
        history_ok: bool,
    ) -> dict[str, float]:
        """
        Compute adaptive signal weights based on data availability.

        Base: dg=45%, market=30%, kalshi=15%, history=10%

        Unavailable signals are zeroed out and the remaining weights are
        re-normalized proportionally so they always sum to 1.0.  This ensures
        the blended probability is on the same scale as the individual signal
        probabilities regardless of which signals are missing.
        """
        w = dict(BASE_WEIGHTS)
        if not kalshi_ok:
            w["kalshi"] = 0.0
        if not market_ok:
            w["market"] = 0.0
        if not history_ok:
            w["history"] = 0.0

        total = sum(w.values())
        if total > 0:
            return {k: v / total for k, v in w.items()}
        return {"dg": 1.0, "market": 0.0, "kalshi": 0.0, "history": 0.0}

    @staticmethod
    def _blend(
        dg_hist: float,
        market: Optional[float],
        kalshi: Optional[float],
        course: Optional[float],
        weights: dict[str, float],
    ) -> float:
        """Weighted blend of available signals."""
        result = weights["dg"] * dg_hist
        if market is not None and weights["market"] > 0:
            result += weights["market"] * market
        if kalshi is not None and weights["kalshi"] > 0:
            result += weights["kalshi"] * kalshi
        if course is not None and weights["history"] > 0:
            result += weights["history"] * course
        return result

    # ── Private: name matching ───────────────────────────────────────────────

    @staticmethod
    def _match_player_name(
        dg_name: str,
        prob_map: dict[str, float],
    ) -> Optional[float]:
        """
        Match a DataGolf "Last, First" name to a prob_map keyed by "First Last".

        Strategy:
          1. Exact normalized match (lowercase, stripped).
          2. Last-name-only fallback — accepted only when exactly one candidate matches.
        """
        if not prob_map:
            return None

        parts = dg_name.split(", ", 1)
        if len(parts) == 2:
            normalized = f"{parts[1]} {parts[0]}".lower().strip()
            last_name = parts[0].lower().strip()
        else:
            normalized = dg_name.lower().strip()
            last_name = normalized

        # Pass 1: exact match
        for name, prob in prob_map.items():
            if name.lower().strip() == normalized:
                return prob

        # Pass 2: last-name fallback
        candidates = [(k, v) for k, v in prob_map.items() if last_name in k.lower()]
        if len(candidates) == 1:
            return candidates[0][1]

        return None

    # ── Private: ranking ─────────────────────────────────────────────────────

    @staticmethod
    def _assign_ranks(players: list[ProprietaryPlayer]) -> list[ProprietaryPlayer]:
        """Assign proprietary_rank, dg_rank, and rank_delta."""
        dg_sorted = sorted(players, key=lambda p: p.dg_win_prob_history, reverse=True)
        for i, p in enumerate(dg_sorted, 1):
            p.dg_rank = i

        prop_sorted = sorted(players, key=lambda p: p.proprietary_win_prob, reverse=True)
        for i, p in enumerate(prop_sorted, 1):
            p.proprietary_rank = i

        for p in players:
            p.rank_delta = p.dg_rank - p.proprietary_rank

        return prop_sorted
