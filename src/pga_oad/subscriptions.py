"""
Subscription tier definitions and feature gating.

Tiers:
  FREE  - Hook them: basic field data, DG rankings, 1 pick tracking
  PRO   - Core value: blended signals, backtesting, season optimizer, pick history
  ELITE - Full power: proprietary model, all 5 signals, leaderboard, priority alerts

Pricing:
  Free  = $0/mo
  Pro   = $14.99/mo or $119.99/yr (save 33%)
  Elite = $29.99/mo or $239.99/yr (save 33%)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TierFeatures:
    """What each subscription tier unlocks."""
    name: str
    monthly_price: float
    yearly_price: float
    # Data access
    field_data: bool = True
    dg_rankings: bool = True
    kalshi_markets: bool = False
    blended_signals: bool = False
    proprietary_model: bool = False
    sportsbook_odds: bool = False
    # Tools
    season_optimizer: bool = False
    backtesting: bool = False
    pick_tracking: bool = True
    pick_history: bool = False
    # Display limits
    field_display_limit: int = 20  # how many players shown in rankings
    alternatives_shown: int = 3   # alt picks shown
    # Social
    leaderboard_access: bool = False
    # Alerts
    weekly_email_alert: bool = False
    # API
    api_access: bool = False
    api_rate_limit: int = 0  # requests per hour


FREE = TierFeatures(
    name="free",
    monthly_price=0.0,
    yearly_price=0.0,
    field_data=True,
    dg_rankings=True,
    kalshi_markets=False,
    blended_signals=False,
    proprietary_model=False,
    sportsbook_odds=False,
    season_optimizer=False,
    backtesting=False,
    pick_tracking=True,
    pick_history=False,
    field_display_limit=20,
    alternatives_shown=3,
    leaderboard_access=False,
    weekly_email_alert=False,
    api_access=False,
    api_rate_limit=10,
)

PRO = TierFeatures(
    name="pro",
    monthly_price=14.99,
    yearly_price=119.99,
    field_data=True,
    dg_rankings=True,
    kalshi_markets=True,
    blended_signals=True,
    proprietary_model=False,
    sportsbook_odds=True,
    season_optimizer=True,
    backtesting=True,
    pick_tracking=True,
    pick_history=True,
    field_display_limit=100,
    alternatives_shown=15,
    leaderboard_access=True,
    weekly_email_alert=True,
    api_access=True,
    api_rate_limit=100,
)

ELITE = TierFeatures(
    name="elite",
    monthly_price=29.99,
    yearly_price=239.99,
    field_data=True,
    dg_rankings=True,
    kalshi_markets=True,
    blended_signals=True,
    proprietary_model=True,
    sportsbook_odds=True,
    season_optimizer=True,
    backtesting=True,
    pick_tracking=True,
    pick_history=True,
    field_display_limit=200,
    alternatives_shown=30,
    leaderboard_access=True,
    weekly_email_alert=True,
    api_access=True,
    api_rate_limit=500,
)

TIERS = {"free": FREE, "pro": PRO, "elite": ELITE}


def get_tier(name: str) -> TierFeatures:
    """Get tier features by name."""
    return TIERS.get(name.lower(), FREE)


def check_feature(tier_name: str, feature: str) -> bool:
    """Check if a feature is available for the given tier."""
    tier = get_tier(tier_name)
    return getattr(tier, feature, False)
