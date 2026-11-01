"""
Database Engine & Session Configuration
Owner: Shivam (Backend & Database)

Provides PostgreSQL and SQLite engine setup controlled through DATABASE_URL.
Defaults to SQLite only when DATABASE_URL is not configured.
Never silently falls back to SQLite when an explicitly configured PostgreSQL connection fails.
"""

from typing import Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.config import settings
from backend.app.db.models import Base

DATABASE_URL = settings.get_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def check_db_connection() -> Tuple[bool, str]:
    """Verify active database connectivity for the /health endpoint."""
    db_dialect = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, f"connected ({db_dialect})"
    except Exception as exc:
        return False, f"disconnected ({db_dialect}): {exc}"


def get_db():
    """FastAPI dependency provider for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
