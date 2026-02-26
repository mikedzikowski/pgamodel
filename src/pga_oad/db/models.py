"""SQLAlchemy ORM models for the subscription platform."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)   # None for OAuth users
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    subscription = relationship("Subscription", back_populates="user", uselist=False)
    picks = relationship("Pick", back_populates="user")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), unique=True, nullable=False)
    tier = Column(Enum("free", "pro", "elite", name="tier_enum"), default="free", nullable=False)
    stripe_customer_id = Column(String(255), nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="subscription")


class Pick(Base):
    """A user's One-and-Done pick for a specific tournament."""
    __tablename__ = "picks"
    __table_args__ = (
        UniqueConstraint("user_id", "event_id", name="uq_user_event_pick"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    event_id = Column(Integer, nullable=False, index=True)
    event_name = Column(String(255), nullable=False)
    dg_id = Column(Integer, nullable=False)
    player_name = Column(String(255), nullable=False)
    # Probabilities at time of pick
    model_win_prob = Column(Float, nullable=True)
    blended_win_prob = Column(Float, nullable=True)
    proprietary_win_prob = Column(Float, nullable=True)
    expected_value = Column(Float, nullable=True)
    picked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    season = Column(Integer, nullable=False)

    user = relationship("User", back_populates="picks")
    result = relationship("PickResult", back_populates="pick", uselist=False)


class PickResult(Base):
    """Actual result for a pick after the tournament completes."""
    __tablename__ = "pick_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pick_id = Column(String(36), ForeignKey("picks.id"), unique=True, nullable=False)
    finish_position = Column(Integer, nullable=True)
    finish_text = Column(String(20), nullable=True)
    made_cut = Column(Boolean, nullable=True)
    score_vs_field = Column(Float, nullable=True)  # percentile rank 0-100
    points_earned = Column(Float, default=0.0)
    resolved_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    pick = relationship("Pick", back_populates="result")


class BacktestRun(Base):
    """Record of a backtesting run."""
    __tablename__ = "backtest_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    season = Column(Integer, nullable=False)
    model_version = Column(String(50), default="proprietary_v1")
    total_events = Column(Integer, default=0)
    total_points = Column(Float, default=0.0)
    avg_finish = Column(Float, nullable=True)
    win_count = Column(Integer, default=0)
    top5_count = Column(Integer, default=0)
    top10_count = Column(Integer, default=0)
    top20_count = Column(Integer, default=0)
    cut_count = Column(Integer, default=0)
    roi_estimate = Column(Float, nullable=True)
    details = Column(Text, nullable=True)  # JSON blob of per-event picks
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
