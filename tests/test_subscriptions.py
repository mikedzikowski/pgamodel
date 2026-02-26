"""Tests for subscription tier logic."""
from __future__ import annotations

from pga_oad.subscriptions import get_tier, check_feature, FREE, PRO, ELITE, TIERS


class TestTierDefinitions:
    def test_free_tier_basics(self):
        assert FREE.name == "free"
        assert FREE.monthly_price == 0.0
        assert FREE.field_data is True
        assert FREE.dg_rankings is True
        assert FREE.proprietary_model is False
        assert FREE.blended_signals is False
        assert FREE.backtesting is False
        assert FREE.field_display_limit == 20

    def test_pro_tier_unlocks(self):
        assert PRO.name == "pro"
        assert PRO.monthly_price == 14.99
        assert PRO.blended_signals is True
        assert PRO.kalshi_markets is True
        assert PRO.backtesting is True
        assert PRO.season_optimizer is True
        assert PRO.proprietary_model is False  # elite only
        assert PRO.field_display_limit == 100

    def test_elite_tier_unlocks_everything(self):
        assert ELITE.name == "elite"
        assert ELITE.monthly_price == 29.99
        assert ELITE.proprietary_model is True
        assert ELITE.blended_signals is True
        assert ELITE.backtesting is True
        assert ELITE.api_rate_limit == 500


class TestFeatureGating:
    def test_get_tier_by_name(self):
        assert get_tier("free") is FREE
        assert get_tier("pro") is PRO
        assert get_tier("elite") is ELITE

    def test_get_tier_unknown_defaults_to_free(self):
        assert get_tier("nonexistent") is FREE

    def test_check_feature(self):
        assert check_feature("free", "field_data") is True
        assert check_feature("free", "backtesting") is False
        assert check_feature("pro", "backtesting") is True
        assert check_feature("pro", "proprietary_model") is False
        assert check_feature("elite", "proprietary_model") is True
