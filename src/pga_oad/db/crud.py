"""Database CRUD operations."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from .models import BacktestRun, Pick, PickResult, Subscription, User


# ── Users ────────────────────────────────────────────────────────────────────

def create_user(db: Session, email: str, username: str, password_hash: Optional[str] = None) -> User:
    user = User(email=email, username=username, password_hash=password_hash)
    db.add(user)
    db.flush()  # ensure user.id is populated before referencing it
    # Create a free subscription automatically
    sub = Subscription(user_id=user.id, tier="free")
    db.add(sub)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


# ── Subscriptions ────────────────────────────────────────────────────────────

def get_subscription(db: Session, user_id: str) -> Optional[Subscription]:
    return db.query(Subscription).filter(Subscription.user_id == user_id).first()


def update_subscription_tier(
    db: Session,
    user_id: str,
    tier: str,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    expires_at: datetime | None = None,
) -> Subscription:
    sub = get_subscription(db, user_id)
    if sub is None:
        sub = Subscription(user_id=user_id, tier=tier)
        db.add(sub)
    else:
        sub.tier = tier
        sub.is_active = True
    if stripe_customer_id:
        sub.stripe_customer_id = stripe_customer_id
    if stripe_subscription_id:
        sub.stripe_subscription_id = stripe_subscription_id
    if expires_at:
        sub.expires_at = expires_at
    db.commit()
    db.refresh(sub)
    return sub


# ── Picks ────────────────────────────────────────────────────────────────────

def create_pick(
    db: Session,
    user_id: str,
    event_id: int,
    event_name: str,
    dg_id: int,
    player_name: str,
    season: int,
    model_win_prob: float | None = None,
    blended_win_prob: float | None = None,
    proprietary_win_prob: float | None = None,
    expected_value: float | None = None,
) -> Pick:
    pick = Pick(
        user_id=user_id,
        event_id=event_id,
        event_name=event_name,
        dg_id=dg_id,
        player_name=player_name,
        season=season,
        model_win_prob=model_win_prob,
        blended_win_prob=blended_win_prob,
        proprietary_win_prob=proprietary_win_prob,
        expected_value=expected_value,
    )
    db.add(pick)
    db.commit()
    db.refresh(pick)
    return pick


def get_user_picks(
    db: Session, user_id: str, season: int | None = None
) -> list[Pick]:
    q = db.query(Pick).filter(Pick.user_id == user_id)
    if season is not None:
        q = q.filter(Pick.season == season)
    return q.order_by(Pick.picked_at.desc()).all()


def get_used_dg_ids(db: Session, user_id: str, season: int) -> set[int]:
    picks = db.query(Pick.dg_id).filter(
        Pick.user_id == user_id, Pick.season == season
    ).all()
    return {p.dg_id for p in picks}


# ── Pick Results ─────────────────────────────────────────────────────────────

def record_pick_result(
    db: Session,
    pick_id: str,
    finish_position: int | None,
    finish_text: str | None,
    made_cut: bool | None,
    score_vs_field: float | None = None,
    points_earned: float = 0.0,
) -> PickResult:
    result = PickResult(
        pick_id=pick_id,
        finish_position=finish_position,
        finish_text=finish_text,
        made_cut=made_cut,
        score_vs_field=score_vs_field,
        points_earned=points_earned,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_user_performance(db: Session, user_id: str, season: int) -> dict:
    """Aggregate performance stats for a user's season."""
    picks = get_user_picks(db, user_id, season)
    total = len(picks)
    resolved = [p for p in picks if p.result is not None]

    if not resolved:
        return {
            "season": season,
            "total_picks": total,
            "resolved_picks": 0,
            "total_points": 0.0,
            "avg_finish": None,
            "wins": 0,
            "top5": 0,
            "top10": 0,
            "top20": 0,
            "cuts": 0,
        }

    total_points = sum(r.result.points_earned for r in resolved)
    finishes = [r.result.finish_position for r in resolved if r.result.finish_position]
    avg_finish = sum(finishes) / len(finishes) if finishes else None

    return {
        "season": season,
        "total_picks": total,
        "resolved_picks": len(resolved),
        "total_points": round(total_points, 2),
        "avg_finish": round(avg_finish, 1) if avg_finish else None,
        "wins": sum(1 for r in resolved if r.result.finish_position == 1),
        "top5": sum(1 for r in resolved if r.result.finish_position and r.result.finish_position <= 5),
        "top10": sum(1 for r in resolved if r.result.finish_position and r.result.finish_position <= 10),
        "top20": sum(1 for r in resolved if r.result.finish_position and r.result.finish_position <= 20),
        "cuts": sum(1 for r in resolved if r.result.made_cut is False),
    }


# ── Backtest ─────────────────────────────────────────────────────────────────

def save_backtest_run(db: Session, run: BacktestRun) -> BacktestRun:
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_backtest_runs(db: Session, season: int | None = None) -> list[BacktestRun]:
    q = db.query(BacktestRun)
    if season is not None:
        q = q.filter(BacktestRun.season == season)
    return q.order_by(BacktestRun.created_at.desc()).all()
