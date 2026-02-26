"""Tests for database models and CRUD operations."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from pga_oad.db.models import Base, User, Pick, PickResult, Subscription, BacktestRun
from pga_oad.db import crud


@pytest.fixture
def db():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestUserCRUD:
    def test_create_user(self, db):
        user = crud.create_user(db, "test@example.com", "testuser", "hashed_pw")
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.is_active is True

    def test_create_user_gets_free_subscription(self, db):
        user = crud.create_user(db, "test@example.com", "testuser", "hashed_pw")
        sub = crud.get_subscription(db, user.id)
        assert sub is not None
        assert sub.tier == "free"
        assert sub.is_active is True

    def test_get_user_by_email(self, db):
        crud.create_user(db, "find@me.com", "findme", "hashed_pw")
        found = crud.get_user_by_email(db, "find@me.com")
        assert found is not None
        assert found.username == "findme"

    def test_get_user_by_email_not_found(self, db):
        assert crud.get_user_by_email(db, "nope@nope.com") is None

    def test_get_user_by_username(self, db):
        crud.create_user(db, "a@b.com", "uniquename", "hashed_pw")
        assert crud.get_user_by_username(db, "uniquename") is not None
        assert crud.get_user_by_username(db, "missing") is None


class TestSubscriptionCRUD:
    def test_upgrade_subscription(self, db):
        user = crud.create_user(db, "pro@user.com", "prouser", "hashed_pw")
        sub = crud.update_subscription_tier(db, user.id, "pro")
        assert sub.tier == "pro"

    def test_upgrade_to_elite(self, db):
        user = crud.create_user(db, "elite@user.com", "eliteuser", "hashed_pw")
        crud.update_subscription_tier(db, user.id, "pro")
        sub = crud.update_subscription_tier(db, user.id, "elite")
        assert sub.tier == "elite"


class TestPickCRUD:
    def test_create_pick(self, db):
        user = crud.create_user(db, "pick@user.com", "picker", "hashed_pw")
        pick = crud.create_pick(
            db,
            user_id=user.id,
            event_id=100,
            event_name="The Masters",
            dg_id=12345,
            player_name="Scheffler, Scottie",
            season=2025,
            model_win_prob=0.15,
        )
        assert pick.player_name == "Scheffler, Scottie"
        assert pick.event_id == 100
        assert pick.season == 2025

    def test_get_used_dg_ids(self, db):
        user = crud.create_user(db, "used@test.com", "usedtest", "hashed_pw")
        crud.create_pick(db, user.id, 1, "Event 1", 111, "Player A", 2025)
        crud.create_pick(db, user.id, 2, "Event 2", 222, "Player B", 2025)

        used = crud.get_used_dg_ids(db, user.id, 2025)
        assert used == {111, 222}

    def test_get_used_dg_ids_different_season(self, db):
        user = crud.create_user(db, "season@test.com", "seasontest", "hashed_pw")
        crud.create_pick(db, user.id, 1, "Event 1", 111, "Player A", 2024)

        used_2025 = crud.get_used_dg_ids(db, user.id, 2025)
        assert used_2025 == set()

    def test_get_user_picks(self, db):
        user = crud.create_user(db, "picks@test.com", "pickstest", "hashed_pw")
        crud.create_pick(db, user.id, 1, "Event 1", 111, "Player A", 2025)
        crud.create_pick(db, user.id, 2, "Event 2", 222, "Player B", 2025)

        picks = crud.get_user_picks(db, user.id, 2025)
        assert len(picks) == 2


class TestPickResultCRUD:
    def test_record_result(self, db):
        user = crud.create_user(db, "result@test.com", "resulttest", "hashed_pw")
        pick = crud.create_pick(db, user.id, 1, "Event 1", 111, "Player A", 2025)

        result = crud.record_pick_result(
            db,
            pick_id=pick.id,
            finish_position=3,
            finish_text="T3",
            made_cut=True,
            score_vs_field=95.0,
            points_earned=15.0,
        )
        assert result.finish_position == 3
        assert result.points_earned == 15.0

    def test_performance_aggregation(self, db):
        user = crud.create_user(db, "perf@test.com", "perftest", "hashed_pw")

        # Create picks with results
        p1 = crud.create_pick(db, user.id, 1, "Event 1", 111, "Player A", 2025)
        crud.record_pick_result(db, p1.id, 1, "1", True, 99.0, 30.0)

        p2 = crud.create_pick(db, user.id, 2, "Event 2", 222, "Player B", 2025)
        crud.record_pick_result(db, p2.id, 15, "T15", True, 60.0, 4.0)

        p3 = crud.create_pick(db, user.id, 3, "Event 3", 333, "Player C", 2025)
        crud.record_pick_result(db, p3.id, None, "CUT", False, 0.0, 0.0)

        perf = crud.get_user_performance(db, user.id, 2025)
        assert perf["total_picks"] == 3
        assert perf["resolved_picks"] == 3
        assert perf["total_points"] == 34.0
        assert perf["wins"] == 1
        assert perf["top5"] == 1
        assert perf["top20"] == 2
        assert perf["cuts"] == 1
