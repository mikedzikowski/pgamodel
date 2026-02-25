from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .models import Prediction


@dataclass
class BlendedPlayer:
    player_name: str
    dg_id: int
    # DataGolf signals
    dg_win_prob: float
    dg_win_prob_history_fit: float
    dg_top10_prob: float
    dg_top20_prob: float
    dg_make_cut_prob: float
    # Kalshi market signals
    kalshi_win_prob: Optional[float] = None
    kalshi_top10_prob: Optional[float] = None
    kalshi_top20_prob: Optional[float] = None
    kalshi_make_cut_prob: Optional[float] = None
    # Blended output
    blended_win_prob: float = 0.0
    dg_rank: int = 0
    kalshi_rank: int = 0
    blended_rank: int = 0
    # Edge: positive means Kalshi market implies higher prob than DataGolf model
    edge: float = 0.0


def _match_player(
    dg_name: str,
    kalshi_probs: dict[str, float],
) -> Optional[float]:
    """
    Match a DataGolf player name to a Kalshi market player name.

    DataGolf format: "Gerard, Ryan"
    Kalshi format:   "Ryan Gerard"

    Tries exact normalized match, then last-name-first-name swap.
    """
    # Convert DataGolf "Last, First" → "First Last"
    parts = dg_name.split(", ", 1)
    if len(parts) == 2:
        dg_normalized = f"{parts[1]} {parts[0]}".lower().strip()
    else:
        dg_normalized = dg_name.lower().strip()

    # Try exact match on normalized name
    for kalshi_name, prob in kalshi_probs.items():
        if kalshi_name.lower().strip() == dg_normalized:
            return prob

    # Try last-name-only match as fallback (handles suffixes like Jr., III)
    last_name = parts[0].lower().strip() if parts else dg_normalized
    candidates = [
        (k, v) for k, v in kalshi_probs.items()
        if last_name in k.lower()
    ]
    if len(candidates) == 1:
        return candidates[0][1]

    return None


class SignalBlender:
    """
    Blends DataGolf model predictions with Kalshi prediction market prices.

    Weights (must sum to 1.0):
        w_dg:     Weight for DataGolf skill model
        w_kalshi: Weight for Kalshi market-implied probability

    When Kalshi data is unavailable for a player, falls back to DataGolf only.
    """

    def __init__(self, w_dg: float = 0.65, w_kalshi: float = 0.35):
        if abs(w_dg + w_kalshi - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.w_dg = w_dg
        self.w_kalshi = w_kalshi

    def blend(
        self,
        predictions: list[Prediction],
        history_fit_map: dict[int, dict],
        kalshi_win_probs: dict[str, float],
        kalshi_top10_probs: dict[str, float] | None = None,
        kalshi_top20_probs: dict[str, float] | None = None,
        kalshi_make_cut_probs: dict[str, float] | None = None,
    ) -> list[BlendedPlayer]:
        """
        Blend DataGolf predictions with Kalshi implied probabilities.

        Args:
            predictions: DataGolf Prediction objects (baseline model)
            history_fit_map: dg_id → raw API dict for baseline_history_fit model
            kalshi_win_probs: player_name → normalized Kalshi win implied prob
            kalshi_top10_probs: player_name → Kalshi top-10 implied prob (optional)
            kalshi_top20_probs: player_name → Kalshi top-20 implied prob (optional)
            kalshi_make_cut_probs: player_name → Kalshi make-cut implied prob (optional)

        Returns:
            List of BlendedPlayer sorted by blended_win_prob descending.
        """
        blended: list[BlendedPlayer] = []

        for pred in predictions:
            hf = history_fit_map.get(pred.dg_id, {})
            dg_win_hf = hf.get("win", pred.win_prob)

            # Prefer history_fit for the DG signal (incorporates course history)
            dg_signal = dg_win_hf

            # Match Kalshi
            k_win = _match_player(pred.player_name, kalshi_win_probs)
            k_top10 = _match_player(pred.player_name, kalshi_top10_probs or {})
            k_top20 = _match_player(pred.player_name, kalshi_top20_probs or {})
            k_cut = _match_player(pred.player_name, kalshi_make_cut_probs or {})

            # Blended win probability
            if k_win is not None:
                blended_win = self.w_dg * dg_signal + self.w_kalshi * k_win
            else:
                blended_win = dg_signal  # fallback: DataGolf only

            edge = (k_win - dg_signal) if k_win is not None else 0.0

            blended.append(
                BlendedPlayer(
                    player_name=pred.player_name,
                    dg_id=pred.dg_id,
                    dg_win_prob=pred.win_prob,
                    dg_win_prob_history_fit=dg_signal,
                    dg_top10_prob=pred.top10_prob,
                    dg_top20_prob=pred.top20_prob,
                    dg_make_cut_prob=pred.make_cut_prob,
                    kalshi_win_prob=k_win,
                    kalshi_top10_prob=k_top10,
                    kalshi_top20_prob=k_top20,
                    kalshi_make_cut_prob=k_cut,
                    blended_win_prob=blended_win,
                    edge=edge,
                )
            )

        # Assign DataGolf ranks
        dg_sorted = sorted(blended, key=lambda p: p.dg_win_prob_history_fit, reverse=True)
        for i, p in enumerate(dg_sorted, 1):
            p.dg_rank = i

        # Assign Kalshi ranks (only for players with Kalshi data)
        k_sorted = sorted(
            [p for p in blended if p.kalshi_win_prob is not None],
            key=lambda p: p.kalshi_win_prob,  # type: ignore[arg-type]
            reverse=True,
        )
        for i, p in enumerate(k_sorted, 1):
            p.kalshi_rank = i

        # Sort by blended win prob and assign final rank
        blended.sort(key=lambda p: p.blended_win_prob, reverse=True)
        for i, p in enumerate(blended, 1):
            p.blended_rank = i

        return blended
