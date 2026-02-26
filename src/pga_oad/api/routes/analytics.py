"""Analytics, backtesting, and performance tracking routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db import crud
from ...db.models import User
from ..deps import get_db, get_current_user, require_tier
from ...services.backtest import BacktestEngine
from ...services.tracker import PerformanceTracker

router = APIRouter(prefix="/analytics", tags=["analytics"])


class PerformanceDashboard(BaseModel):
    """Aggregated performance data for the user dashboard."""
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
    pick_history: list[dict]


class BacktestResult(BaseModel):
    """Result of a backtest run."""
    season: int
    model_version: str
    total_events: int
    total_points: float
    avg_finish: float | None
    win_count: int
    top5_count: int
    top10_count: int
    top20_count: int
    cut_count: int
    roi_estimate: float | None
    event_details: list[dict]


class ModelComparison(BaseModel):
    """Compare model predictions vs actual outcomes."""
    event_name: str
    event_id: int
    model_top_pick: str
    model_win_prob: float
    actual_finish: str | None
    points_earned: float
    better_available: str | None  # best player not yet used


@router.get("/performance/{season}", response_model=PerformanceDashboard)
def get_performance(
    season: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get your season performance dashboard."""
    perf = crud.get_user_performance(db, user.id, season)
    picks = crud.get_user_picks(db, user.id, season)

    pick_history = []
    for p in picks:
        entry = {
            "event_name": p.event_name,
            "player_name": p.player_name,
            "model_win_prob": p.proprietary_win_prob or p.blended_win_prob or p.model_win_prob,
            "picked_at": p.picked_at.isoformat() if p.picked_at else None,
        }
        if p.result:
            entry["finish"] = p.result.finish_text
            entry["points"] = p.result.points_earned
            entry["made_cut"] = p.result.made_cut
        pick_history.append(entry)

    return PerformanceDashboard(
        **perf,
        pick_history=pick_history,
    )


@router.get("/backtest/{season}", response_model=BacktestResult)
def run_backtest(
    season: int,
    user: User = Depends(require_tier("pro")),
    db: Session = Depends(get_db),
    model: str = Query(default="proprietary", description="Model to backtest"),
):
    """Run a historical backtest for the given season (pro tier).

    Simulates what picks the model would have made for each completed
    tournament, then scores them against actual results.
    """
    engine = BacktestEngine()
    result = engine.run(season=season, model_name=model)

    # Save the run
    from ...db.models import BacktestRun
    import json
    run = BacktestRun(
        season=season,
        model_version=model,
        total_events=result["total_events"],
        total_points=result["total_points"],
        avg_finish=result["avg_finish"],
        win_count=result["win_count"],
        top5_count=result["top5_count"],
        top10_count=result["top10_count"],
        top20_count=result["top20_count"],
        cut_count=result["cut_count"],
        roi_estimate=result["roi_estimate"],
        details=json.dumps(result["event_details"]),
    )
    crud.save_backtest_run(db, run)

    return BacktestResult(**result)


@router.get("/leaderboard/{season}")
def get_leaderboard(
    season: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    top: int = Query(default=20, ge=1, le=100),
):
    """Public leaderboard: top users by points this season."""
    # Query all users with picks this season
    from ...db.models import Pick, PickResult
    results = (
        db.query(
            User.username,
            db.query(
                PickResult.points_earned
            ).filter(
                PickResult.pick_id == Pick.id,
                Pick.user_id == User.id,
                Pick.season == season,
            ).correlate(User).label("points"),
        )
        .limit(top)
        .all()
    )

    # Simpler approach: get all picks with results for this season
    picks = (
        db.query(Pick)
        .filter(Pick.season == season)
        .all()
    )

    # Aggregate by user
    user_points: dict[str, dict] = {}
    for pick in picks:
        if pick.result is None:
            continue
        uid = pick.user_id
        if uid not in user_points:
            user_obj = crud.get_user_by_id(db, uid)
            user_points[uid] = {
                "username": user_obj.username if user_obj else "unknown",
                "total_points": 0.0,
                "picks_resolved": 0,
            }
        user_points[uid]["total_points"] += pick.result.points_earned
        user_points[uid]["picks_resolved"] += 1

    leaderboard = sorted(
        user_points.values(), key=lambda x: x["total_points"], reverse=True
    )
    return leaderboard[:top]
