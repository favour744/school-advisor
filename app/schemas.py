"""Request and response models. FastAPI uses these for validation and for /docs."""

from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

CategoryName = Literal[
    "AI_Data",
    "Software_Apps",
    "Systems_Hardware",
    "Programming",
    "Professional",
    "Theory",
    "Networks_Cloud_Security",
]
Band = Literal["At Risk", "Average", "Strong"]
Recommendation = Literal["recommended", "manageable", "high_risk"]
Percent = Annotated[float, Field(ge=0, le=100)]


# ---------------------------------------------------------------- requests
class StudentProfile(BaseModel):
    """Where the student stands right now."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "jamb_score": 230,
                    "study_hours": 11,
                    "attendance": 0.82,
                    "prior_cgpa": 3.7,
                    "prior_semester_gpa": 3.6,
                    "prior_avg_score": 64,
                    "courses_failed_so_far": 0,
                    "courses_taken_so_far": 15,
                    "category_averages": {"Programming": 66, "AI_Data": 61},
                }
            ]
        },
    )

    jamb_score: int = Field(ge=100, le=400, description="JAMB (UTME) score.")
    study_hours: float = Field(ge=0, le=60, description="Study hours per week outside class.")
    attendance: float = Field(ge=0, le=1, description="Share of classes attended, 0 to 1.")
    prior_cgpa: float = Field(ge=0, le=5, description="Current CGPA on the 5-point scale.")
    prior_semester_gpa: float = Field(ge=0, le=5, description="Last semester's GPA, 5-point scale.")
    prior_avg_score: Percent = Field(description="Average of all course scores so far, out of 100.")
    courses_failed_so_far: int = Field(ge=0, le=100)
    courses_taken_so_far: int = Field(ge=0, le=100)
    category_averages: Optional[Dict[CategoryName, Percent]] = Field(
        default=None,
        description=(
            "Optional. Average score in each course area the student has already taken. "
            "Missing areas fall back to prior_avg_score."
        ),
    )
    prereq_avg_score: Optional[Percent] = Field(
        default=None,
        description=(
            "Optional. Average score in the prerequisites for the chosen courses. "
            "Falls back to prior_avg_score."
        ),
    )

    @model_validator(mode="after")
    def failed_not_more_than_taken(self):
        if self.courses_failed_so_far > self.courses_taken_so_far:
            raise ValueError("Courses failed cannot be more than courses taken")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student: StudentProfile
    course_code: str = Field(min_length=1, max_length=40, examples=["CSC 301"])


class RecommendRequest(BaseModel):
    """Send either explicit course_codes, or a level and semester to use that slot's courses."""

    model_config = ConfigDict(extra="forbid")

    student: StudentProfile
    course_codes: Optional[List[str]] = Field(default=None, min_length=1, max_length=20)
    level: Optional[Literal[100, 200, 300, 400]] = None
    semester: Optional[Literal[1, 2]] = None
    include_electives: bool = Field(
        default=False,
        description="Only used with level and semester: also include elective courses.",
    )

    @model_validator(mode="after")
    def need_courses_or_slot(self):
        if self.course_codes is None and (self.level is None or self.semester is None):
            raise ValueError("Send course_codes, or both level and semester")
        return self


# --------------------------------------------------------------- responses
class WhatIf(BaseModel):
    action: str
    predicted_score: float
    change: float


class CoursePrediction(BaseModel):
    course_code: str
    category: str
    level: int
    semester: int
    units: int
    is_elective: bool
    predicted_score: float
    predicted_grade: str
    band: Band
    recommendation: Recommendation
    borderline: bool
    advice: str
    what_if: List[WhatIf]


class Summary(BaseModel):
    course_count: int
    total_units: int
    predicted_gpa: float
    expected_average_score: float
    high_risk_count: int
    message: str


class ModelInfo(BaseModel):
    name: str
    typical_error: float
    r2: Optional[float] = None


class RecommendResponse(BaseModel):
    predictions: List[CoursePrediction]
    summary: Summary
    warnings: List[str]
    model: ModelInfo


class PredictResponse(BaseModel):
    prediction: CoursePrediction
    warnings: List[str]
    model: ModelInfo


class CourseOut(BaseModel):
    course_code: str
    category: str
    level: int
    semester: int
    units: int
    is_elective: bool
    has_prereqs: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    detail: Optional[str] = None
