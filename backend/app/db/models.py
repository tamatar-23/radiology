"""
SQLAlchemy Database Models
Owner: Shivam (Backend & Database)

Defines relational schema models for Scan, Report, and Finding.
Supports both SQLite (local development) and PostgreSQL (Docker environment).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Scan(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # e.g., image/png, image/jpeg
    file_size_bytes = Column(Integer, nullable=True, default=0)
    status = Column(String, nullable=False, default="uploaded")
    image_metadata_json = Column(JSON, nullable=True)
    uploaded_at = Column(DateTime, default=_utc_now)

    reports = relationship("Report", back_populates="scan", cascade="all, delete-orphan")

    @property
    def media_type(self) -> str:
        return self.file_type

    @property
    def stored_path(self) -> str:
        return self.file_path


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id = Column(String, ForeignKey("scans.id"), nullable=False)
    status = Column(String, nullable=False, default="completed")
    execution_mode = Column(String, nullable=False, default="mock")
    model_name = Column(String, nullable=False, default="demo-cxr-densenet121-mock")
    predictions_json = Column(JSON, nullable=True)
    findings_json = Column(JSON, nullable=True)
    impression = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    heatmap_available = Column(Integer, nullable=False, default=0)  # 0 or 1 for SQLite/PG portability
    heatmap_url = Column(String, nullable=True)
    explainability_message = Column(Text, nullable=True)
    evidence_status = Column(String, nullable=False, default="sufficient")
    evidence_json = Column(JSON, nullable=True)
    agent_trace_json = Column(JSON, nullable=True)
    model_metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now)

    scan = relationship("Scan", back_populates="reports")
    findings = relationship("Finding", back_populates="report", cascade="all, delete-orphan")


class Finding(Base):
    __tablename__ = "findings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String, ForeignKey("reports.id"), nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)

    report = relationship("Report", back_populates="findings")
