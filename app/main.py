"""FastAPI application: JSON API under /api plus the web page at /.

Run with:  uvicorn app.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .advisor import Advisor, ModelLoadError, UnknownCourseError
from .schemas import (
    CourseOut,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
)

log = logging.getLogger("advisor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. If it fails, the server still starts and
    reports the reason through /api/health and a 503 on the other endpoints."""
    app.state.advisor = None
    app.state.load_error = None
    try:
        app.state.advisor = Advisor.load()
        log.info("Model loaded: %s", app.state.advisor.model_info()["name"])
    except ModelLoadError as exc:
        app.state.load_error = str(exc)
        log.error("Model not loaded: %s", exc)
    yield


app = FastAPI(
    title="Academic Advisor API",
    version="1.0.0",
    description=(
        "Predicts how a student is likely to score in each course and turns the "
        "predictions into advice. Predictions come from a Ridge regression trained on past "
        "student records and are typically within about 7 points of the real score."
    ),
    lifespan=lifespan,
)


def get_advisor(request: Request) -> Advisor:
    advisor = request.app.state.advisor
    if advisor is None:
        detail = request.app.state.load_error or "The model is not loaded."
        raise HTTPException(status_code=503, detail=detail)
    return advisor


# --------------------------------------------------------------------- API
@app.get("/api/health", response_model=HealthResponse, tags=["service"])
def health(request: Request):
    loaded = request.app.state.advisor is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "detail": None if loaded else request.app.state.load_error,
    }


@app.get("/api/courses", response_model=List[CourseOut], tags=["courses"])
def list_courses(
    level: Optional[int] = Query(default=None, ge=100, le=400),
    semester: Optional[int] = Query(default=None, ge=1, le=2),
    advisor: Advisor = Depends(get_advisor),
):
    """All courses the model knows about, optionally filtered by level and semester."""
    return advisor.course_list(level, semester)


@app.post("/api/predict", response_model=PredictResponse, tags=["advice"])
def predict(body: PredictRequest, advisor: Advisor = Depends(get_advisor)):
    """Prediction and advice for one course."""
    try:
        result = advisor.recommend(body.student.model_dump(), [body.course_code])
    except UnknownCourseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "prediction": result["predictions"][0],
        "warnings": result["warnings"],
        "model": result["model"],
    }


@app.post("/api/recommend", response_model=RecommendResponse, tags=["advice"])
def recommend(body: RecommendRequest, advisor: Advisor = Depends(get_advisor)):
    """Predictions for several courses, ranked from best to worst predicted score."""
    if body.course_codes is not None:
        codes = body.course_codes
    else:
        codes = advisor.slot_codes(body.level, body.semester, body.include_electives)
        if not codes:
            raise HTTPException(
                status_code=422,
                detail=f"No courses found for level {body.level}, semester {body.semester}.",
            )
    try:
        return advisor.recommend(body.student.model_dump(), codes)
    except UnknownCourseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# --------------------------------------------------------------- web page
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
