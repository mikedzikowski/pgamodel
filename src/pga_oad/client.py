from __future__ import annotations
import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from .cache import DataGolfCache

load_dotenv()

BASE_URL = "https://feeds.datagolf.com"


class DataGolfClient:
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/raw"):
        self.api_key = api_key or os.environ["DATAGOLF_API_KEY"]
        self.cache = DataGolfCache(cache_dir)
        self._last_request: float = 0.0

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Any:
        params = params or {}
        params["key"] = self.api_key

        cached = self.cache.get(endpoint, {k: v for k, v in params.items() if k != "key"})
        if cached is not None:
            return cached

        # Respect 1 req/sec rate limit
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        self._last_request = time.time()

        data = response.json()
        cache_params = {k: v for k, v in params.items() if k != "key"}
        self.cache.set(endpoint, cache_params, data)
        return data

    # --- General endpoints ---

    def get_player_list(self) -> dict:
        return self._get("/get-player-list")

    def get_schedule(
        self,
        tour: str = "pga",
        season: Optional[int] = None,
        upcoming_only: bool = False,
    ) -> dict:
        params: dict = {"tour": tour, "upcoming_only": "yes" if upcoming_only else "no"}
        if season is not None:
            params["season"] = season
        return self._get("/get-schedule", params)

    def get_field_updates(self, tour: str = "pga") -> dict:
        return self._get("/field-updates", {"tour": tour})

    # --- Model predictions endpoints ---

    def get_dg_rankings(self) -> dict:
        return self._get("/preds/get-dg-rankings")

    def get_pre_tournament_predictions(
        self,
        tour: str = "pga",
        add_position: Optional[str] = None,
        odds_format: str = "percent",
    ) -> dict:
        params: dict = {"tour": tour, "odds_format": odds_format, "dead_heat": "yes"}
        if add_position:
            params["add_position"] = add_position
        return self._get("/preds/pre-tournament", params)

    def get_pre_tournament_archive(
        self,
        event_id: Optional[int] = None,
        year: int = 2025,
        odds_format: str = "percent",
    ) -> dict:
        params: dict = {"year": year, "odds_format": odds_format}
        if event_id is not None:
            params["event_id"] = event_id
        return self._get("/preds/pre-tournament-archive", params)

    def get_skill_ratings(self, display: str = "value") -> dict:
        return self._get("/preds/skill-ratings", {"display": display})

    def get_player_decompositions(self, tour: str = "pga") -> dict:
        return self._get("/preds/player-decompositions", {"tour": tour})

    # --- Betting tools endpoints ---

    def get_outright_odds(
        self,
        tour: str = "pga",
        market: str = "win",
        odds_format: str = "percent",
    ) -> dict:
        return self._get(
            "/betting-tools/outrights",
            {"tour": tour, "market": market, "odds_format": odds_format},
        )

    def get_matchup_odds(
        self,
        tour: str = "pga",
        market: str = "matchups",
        odds_format: str = "percent",
    ) -> dict:
        return self._get(
            "/betting-tools/matchups",
            {"tour": tour, "market": market, "odds_format": odds_format},
        )

    # --- Historical data endpoints ---

    def get_historical_event_list(self, tour: str = "pga") -> dict:
        return self._get("/historical-event-data/event-list", {"tour": tour})

    def get_historical_events(
        self,
        tour: str = "pga",
        event_id: Optional[int] = None,
        year: Optional[int] = None,
    ) -> dict:
        params: dict = {"tour": tour}
        if event_id is not None:
            params["event_id"] = event_id
        if year is not None:
            params["year"] = year
        return self._get("/historical-event-data/events", params)

    def get_historical_raw_event_list(self, tour: str = "pga") -> dict:
        return self._get("/historical-raw-data/event-list", {"tour": tour})

    def get_historical_rounds(
        self,
        tour: str = "pga",
        event_id: Optional[int] = None,
        year: Optional[int] = None,
    ) -> dict:
        params: dict = {"tour": tour}
        if event_id is not None:
            params["event_id"] = event_id
        if year is not None:
            params["year"] = year
        return self._get("/historical-raw-data/rounds", params)
