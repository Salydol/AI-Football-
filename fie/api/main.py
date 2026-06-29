"""Football Intelligence Engine — FastAPI Application v1.0.0."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from fie.api.routes import tracking as tracking_router
from fie.api.routes import ratings as ratings_router
from fie.api.routes import reports as reports_router
from fie.api.routes import tactical as tactical_router
from fie.api.routes import mistakes as mistakes_router
from fie.api.routes import prediction as prediction_router
from fie.api.routes import analytics as analytics_router
from fie.api.routes import clipping as clipping_router
from fie.api.routes import team_analytics as team_analytics_router
from fie.api.routes import llm as llm_router
from fie.api.routes import academy as academy_router
from fie.api.routes import search as search_router
from fie.api.routes import opponent as opponent_router
from fie.api.routes import explainability as explainability_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FIE v1.0.0 starting up")
    yield
    logger.info("FIE shutting down")


app = FastAPI(
    title="Football Intelligence Engine",
    version="1.0.0",
    description=(
        "AI-powered football analytics: tracking, event detection, "
        "ratings, tactical analysis, video clipping, LLM coaching, "
        "academy tracking, video search by natural language, "
        "opponent weakness scanning, and explainable AI."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tracking_router.router)
app.include_router(ratings_router.router)
app.include_router(reports_router.router)
app.include_router(tactical_router.router)
app.include_router(mistakes_router.router)
app.include_router(prediction_router.router)
app.include_router(analytics_router.router)
app.include_router(clipping_router.router)
app.include_router(team_analytics_router.router)
app.include_router(llm_router.router)
app.include_router(academy_router.router)
app.include_router(search_router.router)
app.include_router(opponent_router.router)
app.include_router(explainability_router.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}
