"""Pick management routes - the core product."""
from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...db import crud
from ...db.models import User
from ..deps import get_db, get_current_user, require_tier

router = APIRouter(prefix="/picks", tags=["picks"])


class MakePickRequest(BaseModel):
    event_id: int
    event_name: str
    dg_id: int
    player_name: str
    season: int
    model_win_prob: float | None = None
    blended_win_prob: float | None = None
    proprietary_win_prob: float | None = None
    expected_value: float | None = None


class PickResponse(BaseModel):
    id: str
    event_id: int
    event_name: str
    dg_id: int
    player_name: str
    season: int
    model_win_prob: float | None
    blended_win_prob: float | None
    proprietary_win_prob: float | None
    expected_value: float | None
    picked_at: str
    result: dict | None = None


class WeeklyRecommendation(BaseModel):
    """The weekly pick recommendation with confidence breakdown."""
    recommended_player: str
    dg_id: int
    event_name: str
    event_id: int
    proprietary_win_prob: float
    blended_win_prob: float
    model_win_prob: float
    expected_value: float
    confidence: str  # "high", "medium", "low"
    edge_vs_market: float
    signals: dict  # breakdown of each signal's contribution
    alternatives: list[dict]  # top 5 alternatives


class SeasonSummary(BaseModel):
    season: int
    total_picks: int
    resolved_picks: int
    total_points: float
    avg_finish: float | None
    wins: int
    top5: int
    top10: int
    top20: int
    cuts: int
    used_players: list[str]


@router.post("/make", response_model=PickResponse)
def make_pick(
    body: MakePickRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lock in a pick for a tournament. Once made, picks cannot be changed."""
    # Check if player already used this season
    used = crud.get_used_dg_ids(db, user.id, body.season)
    if body.dg_id in used:
        raise HTTPException(
            status_code=400,
            detail=f"{body.player_name} already used this season",
        )

    # Check if already picked for this event
    existing = crud.get_user_picks(db, user.id, body.season)
    if any(p.event_id == body.event_id for p in existing):
        raise HTTPException(
            status_code=400,
            detail=f"Already have a pick for event {body.event_name}",
        )

    pick = crud.create_pick(
        db,
        user_id=user.id,
        event_id=body.event_id,
        event_name=body.event_name,
        dg_id=body.dg_id,
        player_name=body.player_name,
        season=body.season,
        model_win_prob=body.model_win_prob,
        blended_win_prob=body.blended_win_prob,
        proprietary_win_prob=body.proprietary_win_prob,
        expected_value=body.expected_value,
    )
    return _pick_to_response(pick)


@router.get("/season/{season}", response_model=list[PickResponse])
def get_season_picks(
    season: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all picks for a season."""
    picks = crud.get_user_picks(db, user.id, season)
    return [_pick_to_response(p) for p in picks]


@router.get("/season/{season}/summary", response_model=SeasonSummary)
def get_season_summary(
    season: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get season performance summary."""
    perf = crud.get_user_performance(db, user.id, season)
    picks = crud.get_user_picks(db, user.id, season)
    return SeasonSummary(
        **perf,
        used_players=[p.player_name for p in picks],
    )


@router.get("/used/{season}", response_model=list[str])
def get_used_players(
    season: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get list of player names already used this season."""
    picks = crud.get_user_picks(db, user.id, season)
    return [p.player_name for p in picks]


def _pick_to_response(pick) -> PickResponse:
    result = None
    if pick.result:
        result = {
            "finish_position": pick.result.finish_position,
            "finish_text": pick.result.finish_text,
            "made_cut": pick.result.made_cut,
            "points_earned": pick.result.points_earned,
            "score_vs_field": pick.result.score_vs_field,
        }
    return PickResponse(
        id=pick.id,
        event_id=pick.event_id,
        event_name=pick.event_name,
        dg_id=pick.dg_id,
        player_name=pick.player_name,
        season=pick.season,
        model_win_prob=pick.model_win_prob,
        blended_win_prob=pick.blended_win_prob,
        proprietary_win_prob=pick.proprietary_win_prob,
        expected_value=pick.expected_value,
        picked_at=pick.picked_at.isoformat() if pick.picked_at else "",
        result=result,
    )
