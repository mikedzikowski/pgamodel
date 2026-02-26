"""SQLAlchemy engine and session management."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_DEFAULT_DB_PATH = Path("data/pga_oad.db")


def get_engine(db_url: str | None = None):
    """Create or return the SQLAlchemy engine."""
    if db_url is None:
        db_path = os.environ.get("PGA_OAD_DB_PATH", str(_DEFAULT_DB_PATH))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{db_path}"
    return create_engine(db_url, echo=False)


def get_session(db_url: str | None = None) -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, auto-closing on exit."""
    engine = get_engine(db_url)
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()


def init_db(db_url: str | None = None) -> None:
    """Create all tables."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
