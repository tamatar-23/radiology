"""
FastAPI Main Application Entrypoint
Owner: Shivam (Backend & Deployment)
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import router as api_router
from backend.app.db.database import init_db

app = FastAPI(
    title="Multimodal Agentic Radiology Assistant API",
    description="Backend API for AI-powered DICOM/image analysis, disease classification, agentic RAG, and radiology reporting.",
    version="0.1.0"
)

# CORS configuration for Next.js frontend communication
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

@app.on_event("startup")
def on_startup():
    # Initialize DB tables if needed
    init_db()

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)
