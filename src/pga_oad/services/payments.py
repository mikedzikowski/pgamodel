"""
Stripe payment integration scaffolding.

This module provides the interface for subscription management via Stripe.
To go live, set these environment variables:
  STRIPE_SECRET_KEY      - Your Stripe secret key
  STRIPE_WEBHOOK_SECRET  - Webhook signing secret
  STRIPE_PRO_PRICE_ID    - Price ID for Pro tier
  STRIPE_ELITE_PRICE_ID  - Price ID for Elite tier

Pricing tiers:
  Free  - Basic field data, DataGolf rankings, 1 pick per week
  Pro   - Blended signals, Kalshi integration, backtesting, optimizer ($14.99/mo)
  Elite - Full proprietary model, all signals, priority alerts ($29.99/mo)
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRO_PRICE_ID = os.environ.get("STRIPE_PRO_PRICE_ID", "")
STRIPE_ELITE_PRICE_ID = os.environ.get("STRIPE_ELITE_PRICE_ID", "")

TIER_PRICES = {
    "pro": STRIPE_PRO_PRICE_ID,
    "elite": STRIPE_ELITE_PRICE_ID,
}


class StripeService:
    """Manages Stripe subscription operations.

    All methods are safe to call without a Stripe key — they return
    placeholder responses for local development.
    """

    def __init__(self):
        self._stripe = None
        if STRIPE_SECRET_KEY:
            try:
                import stripe
                stripe.api_key = STRIPE_SECRET_KEY
                self._stripe = stripe
            except ImportError:
                logger.warning("stripe package not installed; payments disabled")

    @property
    def is_live(self) -> bool:
        return self._stripe is not None

    def create_checkout_session(
        self,
        user_id: str,
        email: str,
        tier: str,
        success_url: str = "http://localhost:8501/success",
        cancel_url: str = "http://localhost:8501/cancel",
    ) -> dict:
        """Create a Stripe Checkout session for subscription signup.

        Returns:
            {"checkout_url": "...", "session_id": "..."}
        """
        if not self.is_live:
            return {
                "checkout_url": f"{success_url}?dev_mode=true&tier={tier}",
                "session_id": "dev_session_placeholder",
                "message": "Stripe not configured — dev mode checkout",
            }

        price_id = TIER_PRICES.get(tier)
        if not price_id:
            return {"error": f"No price configured for tier '{tier}'"}

        session = self._stripe.checkout.Session.create(
            mode="subscription",
            customer_email=email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={"user_id": user_id, "tier": tier},
        )

        return {
            "checkout_url": session.url,
            "session_id": session.id,
        }

    def handle_webhook(self, payload: bytes, sig_header: str) -> dict:
        """Process a Stripe webhook event.

        Returns:
            {"event_type": "...", "user_id": "...", "tier": "...", ...}
        """
        if not self.is_live:
            return {"event_type": "dev_mode", "message": "Stripe not configured"}

        event = self._stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )

        result = {"event_type": event["type"]}

        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            result["user_id"] = session["metadata"].get("user_id")
            result["tier"] = session["metadata"].get("tier")
            result["customer_id"] = session.get("customer")
            result["subscription_id"] = session.get("subscription")

        elif event["type"] == "customer.subscription.deleted":
            sub = event["data"]["object"]
            result["customer_id"] = sub.get("customer")
            result["action"] = "downgrade_to_free"

        return result

    def cancel_subscription(self, stripe_subscription_id: str) -> dict:
        """Cancel a subscription at period end."""
        if not self.is_live:
            return {"status": "dev_mode", "message": "Would cancel subscription"}

        sub = self._stripe.Subscription.modify(
            stripe_subscription_id,
            cancel_at_period_end=True,
        )
        return {"status": "canceling", "cancel_at": sub.get("cancel_at")}
