"""Market data and model prediction routes."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from ...client import DataGolfClient
from ...kalshi import KalshiClient
from ...blend import SignalBlender
from ...db.models import User
from ..deps import get_current_user, require_tier

router = APIRouter(prefix="/markets", tags=["markets"])


class FieldPlayer(BaseModel):
    """Player in the current tournament field - free tier."""
    player_name: str
    dg_id: int
    dg_rank: int
    win_prob: float
    top10_prob: float
    top20_prob: float
    make_cut_prob: float


class BlendedFieldPlayer(BaseModel):
    """Player with blended signals - pro tier."""
    player_name: str
    dg_id: int
    dg_rank: int
    blended_rank: int
    kalshi_rank: int
    dg_win_prob: float
    kalshi_win_prob: float | None
    blended_win_prob: float
    edge: float
    dg_top10_prob: float
    dg_top20_prob: float
    dg_make_cut_prob: float


class ProprietaryFieldPlayer(BaseModel):
    """Player with full proprietary model output - elite tier."""
    player_name: str
    dg_id: int
    proprietary_rank: int
    dg_rank: int
    rank_delta: int
    proprietary_win_prob: float
    dg_win_prob: float
    market_consensus_prob: float | None
    kalshi_win_prob: float | None
    recent_form_score: float | None
    course_history_score: float | None
    weights_used: dict
    finish_history: dict


@router.get("/field", response_model=list[FieldPlayer])
def get_field(
    user: User = Depends(get_current_user),
    top: int = Query(default=30, ge=1, le=200),
):
    """Get current tournament field with DataGolf probabilities (free tier)."""
    client = DataGolfClient()
    data = client.get_pre_tournament_predictions(tour="pga", odds_format="percent")
    baseline = data.get("baseline", [])

    players = []
    for i, p in enumerate(
        sorted(baseline, key=lambda x: float(x.get("win", 0)), reverse=True), 1
    ):
        players.append(FieldPlayer(
            player_name=p["player_name"],
            dg_id=p["dg_id"],
            dg_rank=i,
            win_prob=round(float(p.get("win", 0)) * 100, 2),
            top10_prob=round(float(p.get("top_10", 0)) * 100, 2),
            top20_prob=round(float(p.get("top_20", 0)) * 100, 2),
            make_cut_prob=round(float(p.get("make_cut", 0)) * 100, 2),
        ))
        if len(players) >= top:
            break

    return players


@router.get("/field/blended", response_model=list[BlendedFieldPlayer])
def get_blended_field(
    user: User = Depends(require_tier("pro")),
    w_dg: float = Query(default=0.65, ge=0.0, le=1.0),
    w_kalshi: float = Query(default=0.35, ge=0.0, le=1.0),
    top: int = Query(default=50, ge=1, le=200),
):
    """Get blended field rankings (pro tier) - DataGolf + Kalshi signals."""
    client = DataGolfClient()
    kalshi = KalshiClient()
    blender = SignalBlender(w_dg=w_dg, w_kalshi=w_kalshi)

    data = client.get_pre_tournament_predictions(tour="pga", odds_format="percent")

    from ...models import Prediction
    baseline = data.get("baseline", [])
    history_fit = data.get("baseline_history_fit", [])
    hf_map = {p["dg_id"]: p for p in history_fit}

    predictions = []
    for p in baseline:
        predictions.append(Prediction(
            player_name=p["player_name"],
            dg_id=p["dg_id"],
            win_prob=float(p.get("win", 0)),
            top5_prob=float(p.get("top_5", 0)),
            top10_prob=float(p.get("top_10", 0)),
            top20_prob=float(p.get("top_20", 0)),
            make_cut_prob=float(p.get("make_cut", 0)),
        ))

    # Fetch Kalshi markets
    try:
        markets = kalshi.get_active_pga_markets()
        event_code = markets[0]["event_ticker"].rsplit("-", 1)[0] if markets else ""
        kalshi_win = kalshi.get_implied_probs(event_code, "win", normalize=True) if event_code else {}
    except Exception:
        kalshi_win = {}

    blended = blender.blend(predictions, hf_map, kalshi_win)

    result = []
    for bp in blended[:top]:
        result.append(BlendedFieldPlayer(
            player_name=bp.player_name,
            dg_id=bp.dg_id,
            dg_rank=bp.dg_rank,
            blended_rank=bp.blended_rank,
            kalshi_rank=bp.kalshi_rank,
            dg_win_prob=round(bp.dg_win_prob * 100, 2),
            kalshi_win_prob=round(bp.kalshi_win_prob * 100, 2) if bp.kalshi_win_prob else None,
            blended_win_prob=round(bp.blended_win_prob * 100, 2),
            edge=round(bp.edge * 100, 2),
            dg_top10_prob=round(bp.dg_top10_prob * 100, 2),
            dg_top20_prob=round(bp.dg_top20_prob * 100, 2),
            dg_make_cut_prob=round(bp.dg_make_cut_prob * 100, 2),
        ))

    return result


@router.get("/schedule")
def get_schedule(
    user: User = Depends(get_current_user),
    season: int = Query(default=2025),
    upcoming_only: bool = Query(default=False),
):
    """Get PGA Tour schedule."""
    client = DataGolfClient()
    return client.get_schedule(tour="pga", season=season, upcoming_only=upcoming_only)
