"""
Database Engine & Session Configuration
Owner: Shivam (Backend & Database)

Provides PostgreSQL engine setup with fallback to SQLite for local development.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.app.db.models import Base

# Fetch DATABASE_URL from environment; fallback to sqlite memory DB for immediate runnable testing
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./radiology_dev.db")

# Handle SQLite connect args if needed
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency provider for database session management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
