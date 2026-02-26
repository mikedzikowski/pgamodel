"""Tests for the backtesting engine."""
from __future__ import annotations

import pytest

from pga_oad.services.backtest import BacktestEngine, POINTS


class TestScoring:
    """Test the scoring/parsing methods of BacktestEngine."""

    @pytest.fixture
    def engine(self):
        return BacktestEngine.__new__(BacktestEngine)

    def test_parse_win(self, engine):
        assert engine._parse_finish("1") == 1

    def test_parse_tied_position(self, engine):
        assert engine._parse_finish("T5") == 5
        assert engine._parse_finish("T12") == 12

    def test_parse_cut(self, engine):
        assert engine._parse_finish("CUT") is None

    def test_parse_wd(self, engine):
        assert engine._parse_finish("WD") is None

    def test_parse_empty(self, engine):
        assert engine._parse_finish("") is None
        assert engine._parse_finish("-") is None

    def test_score_win(self, engine):
        assert engine._score_finish(1, "1") == POINTS["win"]

    def test_score_top5(self, engine):
        assert engine._score_finish(3, "T3") == POINTS["top5"]
        assert engine._score_finish(5, "T5") == POINTS["top5"]

    def test_score_top10(self, engine):
        assert engine._score_finish(7, "T7") == POINTS["top10"]

    def test_score_top20(self, engine):
        assert engine._score_finish(15, "T15") == POINTS["top20"]

    def test_score_made_cut(self, engine):
        assert engine._score_finish(45, "T45") == POINTS["made_cut"]

    def test_score_missed_cut(self, engine):
        assert engine._score_finish(None, "CUT") == POINTS["missed_cut"]

    def test_aggregation_empty(self, engine):
        result = engine._aggregate(2024, "test", [])
        assert result["total_events"] == 0
        assert result["total_points"] == 0.0

    def test_aggregation_with_data(self, engine):
        details = [
            {"event_id": 1, "event_name": "E1", "player_name": "P1",
             "dg_id": 1, "pre_tournament_win_prob": 10.0,
             "finish_text": "1", "finish_position": 1, "points_earned": 30.0},
            {"event_id": 2, "event_name": "E2", "player_name": "P2",
             "dg_id": 2, "pre_tournament_win_prob": 8.0,
             "finish_text": "T8", "finish_position": 8, "points_earned": 8.0},
            {"event_id": 3, "event_name": "E3", "player_name": "P3",
             "dg_id": 3, "pre_tournament_win_prob": 5.0,
             "finish_text": "CUT", "finish_position": None, "points_earned": 0.0},
        ]
        result = engine._aggregate(2024, "test", details)
        assert result["total_events"] == 3
        assert result["total_points"] == 38.0
        assert result["win_count"] == 1
        assert result["top10_count"] == 2
        assert result["cut_count"] == 1
        assert result["avg_finish"] == 4.5  # (1 + 8) / 2
