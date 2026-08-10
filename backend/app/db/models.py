"""
SQLAlchemy Database Models
Owner: Shivam (Backend & Database)

Defines PostgreSQL schema models for Scan, Report, and Finding.
"""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Scan(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # DICOM, PNG, JPEG
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    reports = relationship("Report", back_populates="scan", cascade="all, delete-orphan")


class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    scan_id = Column(String, ForeignKey("scans.id"), nullable=False)
    impression = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    evidence_json = Column(JSON, nullable=True)  # List of retrieved evidence citations
    created_at = Column(DateTime, default=datetime.utcnow)

    scan = relationship("Scan", back_populates="reports")
    findings = relationship("Finding", back_populates="report", cascade="all, delete-orphan")


class Finding(Base):
    __tablename__ = "findings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String, ForeignKey("reports.id"), nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)

    report = relationship("Report", back_populates="findings")
