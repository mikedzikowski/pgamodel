from __future__ import annotations
import re
import time
from typing import Optional

import requests

from .cache import DataGolfCache

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Golf win/finish series available on Kalshi
GOLF_SERIES = {
    "win": "KXPGATOUR",
    "top5": "KXPGATOP5",
    "top10": "KXPGATOP10",
    "top20": "KXPGATOP20",
    "make_cut": "KXPGAMAKECUT",
}


class KalshiClient:
    """Read-only Kalshi client for golf prediction market data. No auth required."""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache = DataGolfCache(cache_dir)
        self._last_request: float = 0.0

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        params = params or {}

        cached = self.cache.get(f"kalshi_{endpoint}", params)
        if cached is not None:
            return cached

        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)

        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        response = requests.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        self._last_request = time.time()

        data = response.json()
        self.cache.set(f"kalshi_{endpoint}", params, data)
        return data

    def _get_all_pages(self, endpoint: str, result_key: str, params: Optional[dict] = None) -> list:
        """Fetch all pages from a paginated Kalshi endpoint."""
        params = dict(params or {})
        params["limit"] = 1000
        results = []

        while True:
            data = self._get(endpoint, dict(params))
            results.extend(data.get(result_key, []))
            cursor = data.get("cursor")
            if not cursor:
                break
            params["cursor"] = cursor

        return results

    def detect_current_event_code(self) -> Optional[str]:
        """
        Auto-detect the current PGA Tour event code from open Kalshi win markets.
        Event codes look like 'COCITPB26' (Cognizant Classic 2026).
        """
        markets = self._get_all_pages(
            "markets",
            "markets",
            {"series_ticker": GOLF_SERIES["win"], "status": "open"},
        )
        if not markets:
            return None
        # Extract event code from ticker: KXPGATOUR-COCITPB26-WZAL → COCITPB26
        ticker = markets[0].get("ticker", "")
        parts = ticker.split("-")
        if len(parts) >= 2:
            return parts[1]
        return None

    def _extract_player_name(self, title: str) -> Optional[str]:
        """Extract player name from Kalshi market title: 'Will {Name} win the ...'"""
        match = re.match(r"Will (.+?) (?:win|finish)", title)
        if match:
            return match.group(1).strip()
        return None

    def _normalize_name(self, name: str) -> str:
        """Normalize to lowercase stripped for fuzzy matching."""
        return name.lower().strip()

    def get_market_data(
        self,
        event_code: str,
        market: str = "win",
    ) -> list[dict]:
        """
        Fetch all player markets for a given event and market type.

        Args:
            event_code: Event code like 'COCITPB26'
            market: One of 'win', 'top5', 'top10', 'top20', 'make_cut'

        Returns:
            List of market dicts with player_name, yes_bid, yes_ask, last_price, implied_prob
        """
        series = GOLF_SERIES.get(market)
        if not series:
            raise ValueError(f"Unknown market type '{market}'. Choose from: {list(GOLF_SERIES)}")

        raw_markets = self._get_all_pages(
            "markets",
            "markets",
            {"series_ticker": series, "status": "open"},
        )

        # Filter to this event and parse
        results = []
        for m in raw_markets:
            ticker = m.get("ticker", "")
            if event_code not in ticker:
                continue

            title = m.get("title", "")
            player_name = self._extract_player_name(title)
            if not player_name:
                continue

            yes_bid = m.get("yes_bid", 0) or 0
            yes_ask = m.get("yes_ask", 0) or 0
            last_price = m.get("last_price", 0) or 0

            # Use mid-market if both sides available, else last price, else bid
            if yes_bid > 0 and yes_ask > 0:
                raw_prob = (yes_bid + yes_ask) / 2 / 100
            elif last_price > 0:
                raw_prob = last_price / 100
            elif yes_bid > 0:
                raw_prob = yes_bid / 100
            else:
                raw_prob = 0.0

            # Thin market: no real trading activity (floor price artifact)
            thin = yes_bid == 0 and yes_ask == 0 and last_price <= 1

            results.append({
                "ticker": ticker,
                "player_name": player_name,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "last_price": last_price,
                "raw_prob": raw_prob,
                "thin": thin,
            })

        return results

    def get_implied_probs(
        self,
        event_code: str,
        market: str = "win",
        normalize: bool = True,
    ) -> dict[str, float]:
        """
        Return a dict mapping player_name (normalized) → implied probability.

        Probabilities are normalized so they sum to 1.0 (removing the vig).

        Args:
            event_code: Current event code
            market: Market type ('win', 'top5', 'top10', 'top20', 'make_cut')
            normalize: Whether to normalize probabilities to sum to 1.0
        """
        market_data = self.get_market_data(event_code, market)
        if not market_data:
            return {}

        # Exclude thin markets from probability calculations (floor price artifacts)
        liquid = [m for m in market_data if not m.get("thin", False)]
        if not liquid:
            liquid = market_data  # fallback: use all if everything is thin

        raw = {m["player_name"]: m["raw_prob"] for m in liquid}

        if normalize:
            total = sum(raw.values())
            if total > 0:
                return {name: prob / total for name, prob in raw.items()}

        return raw

    def get_thin_players(self, event_code: str, market: str = "win") -> set[str]:
        """Return set of player names with thin (illiquid) markets."""
        return {
            m["player_name"]
            for m in self.get_market_data(event_code, market)
            if m.get("thin", False)
        }

    def get_matchups(self, event_code: str) -> list[dict]:
        """Fetch head-to-head matchup markets for the current event."""
        raw_markets = self._get_all_pages(
            "markets",
            "markets",
            {"series_ticker": "KXPGAH2H", "status": "open"},
        )
        results = []
        for m in raw_markets:
            if event_code not in m.get("ticker", ""):
                continue
            yes_bid = m.get("yes_bid", 0) or 0
            last_price = m.get("last_price", 0) or 0
            implied = (yes_bid or last_price) / 100
            results.append({
                "ticker": m.get("ticker", ""),
                "title": m.get("title", ""),
                "implied_prob": implied,
                "yes_bid": yes_bid,
                "last_price": last_price,
            })
        return results
