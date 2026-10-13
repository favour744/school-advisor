"""Paths and constants shared by the app.

Anything that encodes a decision (grade cut-offs, band thresholds, how big the
"what if" nudges are) lives here so it can be changed in one place.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths can be overridden with environment variables when deploying.
MODEL_PATH = Path(os.getenv("ADVISOR_MODEL_PATH", BASE_DIR / "model" / "advisor_model.pkl"))
COURSES_PATH = Path(os.getenv("ADVISOR_COURSES_PATH", BASE_DIR / "model" / "course_info.csv"))
METRICS_PATH = Path(os.getenv("ADVISOR_METRICS_PATH", BASE_DIR / "model" / "metrics.json"))
STATIC_DIR = BASE_DIR / "app" / "static"

# Columns the model was trained on (same as X.columns in the Colab notebook).
FEATURE_COLUMNS = [
    "course_code",
    "jamb_score",
    "level",
    "semester",
    "units",
    "is_elective",
    "category",
    "study_hours",
    "attendance",
    "prior_cgpa",
    "prior_semester_gpa",
    "prior_avg_score",
    "courses_failed_so_far",
    "courses_taken_so_far",
    "category_avg_prior",
    "has_category_history",
    "prereq_avg_score",
    "has_prereqs",
]

CATEGORIES = [
    "AI_Data",
    "Software_Apps",
    "Systems_Hardware",
    "Programming",
    "Professional",
    "Theory",
    "Networks_Cloud_Security",
]

CATEGORY_LABELS = {
    "AI_Data": "AI and data",
    "Software_Apps": "software and apps",
    "Systems_Hardware": "systems and hardware",
    "Programming": "programming",
    "Professional": "professional",
    "Theory": "theory",
    "Networks_Cloud_Security": "networks, cloud and security",
}

# Grade scale used in the dataset (score cut-off, letter, grade points on a 5-point scale).
GRADE_SCALE = [
    (70.0, "A", 5),
    (60.0, "B", 4),
    (50.0, "C", 3),
    (45.0, "D", 2),
    (40.0, "E", 1),
    (0.0, "F", 0),
]

# Bands used in the dataset's grade_band column.
AT_RISK_BELOW = 50.0
STRONG_FROM = 65.0

# Test-set error of the best model in the notebook (Ridge, MAE 6.644). Used when
# model/metrics.json is not present.
DEFAULT_TYPICAL_ERROR = 6.6

# A prediction this close to a band boundary is flagged as borderline.
BORDERLINE_MARGIN = 4.0

# Size of the nudges used for the "what if" suggestions.
ATTENDANCE_STEP = 0.10  # 10 percentage points
STUDY_HOURS_STEP = 3.0  # hours per week
MAX_STUDY_HOURS = 60.0

# Ranges seen in the training data. Inputs outside them get a warning.
TRAINING_RANGES = {
    "jamb_score": (160, 330, "JAMB score"),
    "study_hours": (1.0, 24.9, "Study hours per week"),
    "attendance": (0.40, 1.00, "Attendance"),
    "prior_avg_score": (30.8, 94.8, "Average score"),
}
