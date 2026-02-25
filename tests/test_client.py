import pytest
from unittest.mock import patch, MagicMock
from pga_oad.client import DataGolfClient
from pga_oad.models import Prediction, Tournament
from pga_oad.optimizer import OneAndDoneOptimizer
from pga_oad.cache import DataGolfCache
import tempfile
import os


@pytest.fixture
def tmp_cache(tmp_path):
    return DataGolfCache(cache_dir=tmp_path)


@pytest.fixture
def client(tmp_path):
    return DataGolfClient(api_key="test_key", cache_dir=str(tmp_path))


class TestDataGolfCache:
    def test_miss_returns_none(self, tmp_cache):
        result = tmp_cache.get("/test-endpoint", {"param": "value"})
        assert result is None

    def test_set_then_get(self, tmp_cache):
        data = {"players": [{"name": "Tiger Woods"}]}
        tmp_cache.set("/test-endpoint", {"param": "value"}, data)
        result = tmp_cache.get("/test-endpoint", {"param": "value"})
        assert result == data

    def test_different_params_are_different_keys(self, tmp_cache):
        tmp_cache.set("/endpoint", {"a": 1}, {"data": "first"})
        tmp_cache.set("/endpoint", {"a": 2}, {"data": "second"})
        assert tmp_cache.get("/endpoint", {"a": 1}) == {"data": "first"}
        assert tmp_cache.get("/endpoint", {"a": 2}) == {"data": "second"}


class TestDataGolfClient:
    def test_get_uses_cache_on_second_call(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"schedule": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            client.get_schedule(tour="pga")
            client.get_schedule(tour="pga")
            assert mock_get.call_count == 1  # second call served from cache

    def test_api_key_not_cached(self, client, tmp_path):
        mock_response = MagicMock()
        mock_response.json.return_value = {"players": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            client.get_player_list()

        # Check cache files don't contain the API key
        for cache_file in tmp_path.glob("*.json"):
            content = cache_file.read_text()
            assert "test_key" not in content


class TestOneAndDoneOptimizer:
    def _make_predictions(self, n=5) -> list[Prediction]:
        return [
            Prediction(
                player_name=f"Player {i}",
                dg_id=i,
                win_prob=0.05 + i * 0.01,
                top5_prob=0.15 + i * 0.02,
                top10_prob=0.25 + i * 0.02,
                top20_prob=0.40 + i * 0.02,
                make_cut_prob=0.70,
            )
            for i in range(1, n + 1)
        ]

    def test_one_pick_per_tournament(self):
        tournaments = [
            Tournament(event_id=1, event_name="Event A", date="2026-03-01"),
            Tournament(event_id=2, event_name="Event B", date="2026-03-08"),
        ]
        preds = self._make_predictions(5)
        predictions = {1: preds, 2: preds}

        optimizer = OneAndDoneOptimizer()
        picks = optimizer.optimize(tournaments, predictions)

        assert len(picks) == 2

    def test_no_repeated_players(self):
        tournaments = [
            Tournament(event_id=1, event_name="Event A", date="2026-03-01"),
            Tournament(event_id=2, event_name="Event B", date="2026-03-08"),
        ]
        preds = self._make_predictions(5)
        predictions = {1: preds, 2: preds}

        optimizer = OneAndDoneOptimizer()
        picks = optimizer.optimize(tournaments, predictions)

        player_ids = [p.dg_id for p in picks]
        assert len(player_ids) == len(set(player_ids))

    def test_used_players_excluded(self):
        tournaments = [
            Tournament(event_id=1, event_name="Event A", date="2026-03-01"),
        ]
        preds = self._make_predictions(5)
        # Mark all but player 3 as used
        used = {1, 2, 4, 5}
        predictions = {1: preds}

        optimizer = OneAndDoneOptimizer(used_players=used)
        picks = optimizer.optimize(tournaments, predictions)

        assert len(picks) == 1
        assert picks[0].dg_id == 3

    def test_picks_ordered_by_tournament(self):
        tournaments = [
            Tournament(event_id=10, event_name="First", date="2026-03-01"),
            Tournament(event_id=20, event_name="Second", date="2026-03-08"),
            Tournament(event_id=30, event_name="Third", date="2026-03-15"),
        ]
        preds = self._make_predictions(10)
        predictions = {10: preds, 20: preds, 30: preds}

        optimizer = OneAndDoneOptimizer()
        picks = optimizer.optimize(tournaments, predictions)

        event_ids = [p.event_id for p in picks]
        assert event_ids == [10, 20, 30]
