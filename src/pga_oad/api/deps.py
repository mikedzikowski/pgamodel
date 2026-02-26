"""FastAPI dependency injection."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..db.engine import get_engine
from ..db.models import User, Subscription
from ..db import crud
from .auth import decode_access_token

from sqlalchemy.orm import sessionmaker

_security = HTTPBearer()


def get_db():
    """Yield a database session."""
    engine = get_engine()
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate the current user from JWT."""
    payload = decode_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    user = crud.get_user_by_id(db, payload["sub"])
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


def require_tier(minimum: str):
    """Dependency factory that checks subscription tier.

    Tier hierarchy: free < pro < elite
    """
    tier_rank = {"free": 0, "pro": 1, "elite": 2}

    def checker(
        user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> User:
        sub = crud.get_subscription(db, user.id)
        user_tier = sub.tier if sub and sub.is_active else "free"
        if tier_rank.get(user_tier, 0) < tier_rank.get(minimum, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires a '{minimum}' subscription. "
                       f"Your current tier: '{user_tier}'.",
            )
        return user

    return checker
