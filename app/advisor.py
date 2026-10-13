"""The advisor: turns a student profile plus course codes into predictions and advice.

No web code lives here, so it can be tested and reused on its own.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

REQUIRED_COURSE_COLUMNS = {"course_code", "category", "units", "is_elective", "level", "semester"}


class ModelLoadError(RuntimeError):
    """The model or course table could not be loaded. The message is safe to show to the user."""


class UnknownCourseError(ValueError):
    def __init__(self, codes: List[str]):
        self.codes = codes
        super().__init__(
            "Unknown course code(s): " + ", ".join(codes) + ". Use GET /api/courses for valid codes."
        )


# ------------------------------------------------------------------ helpers
def grade_for(score: float) -> Tuple[str, int]:
    """Letter grade and grade points for a score."""
    for cutoff, letter, points in config.GRADE_SCALE:
        if score >= cutoff:
            return letter, points
    return "F", 0


def band_for(score: float) -> str:
    if score < config.AT_RISK_BELOW:
        return "At Risk"
    if score < config.STRONG_FROM:
        return "Average"
    return "Strong"


_RECOMMENDATION = {"Strong": "recommended", "Average": "manageable", "At Risk": "high_risk"}


def _is_borderline(score: float) -> bool:
    margin = config.BORDERLINE_MARGIN
    return (
        abs(score - config.AT_RISK_BELOW) < margin or abs(score - config.STRONG_FROM) < margin
    )


def _advice(band: str, borderline: bool) -> str:
    text = {
        "Strong": "A good fit for your record. Keep up your current routine.",
        "Average": "Within reach, but not a sure thing. Steady attendance and study time will protect your result.",
        "At Risk": (
            "Predicted to be hard for you. Pair it with lighter courses, attend every class "
            "and start revising early."
        ),
    }[band]
    if borderline:
        text += " The prediction is close to a band boundary, so it could go either way."
    return text


# ------------------------------------------------------------------ advisor
class Advisor:
    def __init__(self, model, courses: pd.DataFrame, metrics: Optional[dict] = None):
        self.model = model
        self.metrics = metrics or {}

        if courses["course_code"].duplicated().any():
            raise ModelLoadError("course_info.csv has duplicate course codes.")
        self.courses = courses.set_index("course_code", drop=False)

        # Use the exact columns the model was fitted on when it records them.
        fitted = getattr(model, "feature_names_in_", None)
        self.features = list(fitted) if fitted is not None else list(config.FEATURE_COLUMNS)
        unsupported = sorted(set(self.features) - set(config.FEATURE_COLUMNS))
        if unsupported:
            raise ModelLoadError(
                "The model expects columns this app does not provide: " + ", ".join(unsupported)
            )

    # ---- loading
    @classmethod
    def load(
        cls,
        model_path: Optional[Path] = None,
        courses_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ) -> "Advisor":
        model_path = Path(model_path or config.MODEL_PATH)
        courses_path = Path(courses_path or config.COURSES_PATH)
        metrics_path = Path(metrics_path or config.METRICS_PATH)

        if not model_path.exists():
            raise ModelLoadError(
                f"Model file not found: {model_path}. Put advisor_model.pkl from your Colab "
                "notebook in the model/ folder, or run `python train_model.py` to build one."
            )
        if not courses_path.exists():
            raise ModelLoadError(
                f"Course table not found: {courses_path}. Put course_info.csv in the model/ folder."
            )

        try:
            model = joblib.load(model_path)
        except Exception as exc:  # unpickling can fail in many ways
            raise ModelLoadError(
                f"Could not load {model_path.name}: {exc}. Check that scikit-learn matches the "
                "version used in Colab (requirements.txt pins 1.6.1)."
            ) from exc

        courses = pd.read_csv(courses_path)
        missing = REQUIRED_COURSE_COLUMNS - set(courses.columns)
        if missing:
            raise ModelLoadError(
                f"course_info.csv is missing columns: {', '.join(sorted(missing))}."
            )
        if "has_prereqs" not in courses.columns:
            log.warning(
                "course_info.csv has no has_prereqs column; assuming every course has prerequisites. "
                "Re-run train_model.py to regenerate it."
            )
            courses["has_prereqs"] = 1

        metrics = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except (OSError, ValueError):
                log.warning("Could not read %s; using default error estimate.", metrics_path)

        return cls(model, courses, metrics)

    # ---- model info
    @property
    def typical_error(self) -> float:
        return float(self.metrics.get("mae", config.DEFAULT_TYPICAL_ERROR))

    def model_info(self) -> dict:
        name = self.metrics.get("model")
        if not name:
            final = getattr(self.model, "steps", [("", self.model)])[-1][1]
            name = type(final).__name__
        r2 = self.metrics.get("r2")
        return {
            "name": str(name),
            "typical_error": round(self.typical_error, 1),
            "r2": None if r2 is None else round(float(r2), 3),
        }

    # ---- courses
    def course_list(self, level: Optional[int] = None, semester: Optional[int] = None) -> List[dict]:
        df = self.courses
        if level is not None:
            df = df[df["level"] == level]
        if semester is not None:
            df = df[df["semester"] == semester]
        # The index is also named course_code; drop it so sorting by the column is unambiguous.
        df = df.reset_index(drop=True).sort_values(["level", "semester", "is_elective", "course_code"])
        return [self._course_dict(row) for _, row in df.iterrows()]

    @staticmethod
    def _course_dict(row) -> dict:
        return {
            "course_code": str(row["course_code"]),
            "category": str(row["category"]),
            "level": int(row["level"]),
            "semester": int(row["semester"]),
            "units": int(row["units"]),
            "is_elective": bool(int(row["is_elective"])),
            "has_prereqs": bool(int(row["has_prereqs"])),
        }

    def slot_codes(self, level: int, semester: int, include_electives: bool) -> List[str]:
        rows = self.course_list(level, semester)
        return [c["course_code"] for c in rows if include_electives or not c["is_elective"]]

    # ---- prediction
    def _build_frame(self, student: dict, codes: List[str]) -> pd.DataFrame:
        category_averages = student.get("category_averages") or {}
        prereq_given = student.get("prereq_avg_score")
        overall = float(student["prior_avg_score"])

        rows = []
        for code in codes:
            course = self.courses.loc[code]
            category = str(course["category"])
            has_prereqs = int(course["has_prereqs"])

            # Mirrors how the training data was built: with no history in a
            # category (or no prerequisites), the average falls back to the overall one.
            if category in category_averages:
                category_avg, has_history = float(category_averages[category]), 1
            else:
                category_avg, has_history = overall, 0
            prereq_avg = float(prereq_given) if (has_prereqs and prereq_given is not None) else overall

            rows.append(
                {
                    "course_code": code,
                    "jamb_score": int(student["jamb_score"]),
                    "level": int(course["level"]),
                    "semester": int(course["semester"]),
                    "units": int(course["units"]),
                    "is_elective": int(course["is_elective"]),
                    "category": category,
                    "study_hours": float(student["study_hours"]),
                    "attendance": float(student["attendance"]),
                    "prior_cgpa": float(student["prior_cgpa"]),
                    "prior_semester_gpa": float(student["prior_semester_gpa"]),
                    "prior_avg_score": overall,
                    "courses_failed_so_far": int(student["courses_failed_so_far"]),
                    "courses_taken_so_far": int(student["courses_taken_so_far"]),
                    "category_avg_prior": category_avg,
                    "has_category_history": has_history,
                    "prereq_avg_score": prereq_avg,
                    "has_prereqs": has_prereqs,
                }
            )
        return pd.DataFrame(rows, columns=self.features)

    def predict_scores(self, student: dict, codes: List[str]) -> np.ndarray:
        frame = self._build_frame(student, codes)
        raw = np.asarray(self.model.predict(frame), dtype=float)
        return np.clip(raw, 0.0, 100.0)

    # ---- what-if scenarios
    def _scenarios(self, student: dict) -> List[Tuple[str, dict]]:
        scenarios = []

        attendance = float(student["attendance"])
        better_attendance = min(1.0, attendance + config.ATTENDANCE_STEP)
        if better_attendance - attendance >= 0.005:
            scenarios.append(
                (
                    f"Attend more classes: from {attendance * 100:.0f}% to {better_attendance * 100:.0f}%",
                    {**student, "attendance": better_attendance},
                )
            )

        hours = float(student["study_hours"])
        more_hours = min(config.MAX_STUDY_HOURS, hours + config.STUDY_HOURS_STEP)
        if more_hours - hours >= 0.5:
            scenarios.append(
                (
                    f"Study {more_hours - hours:g} more hours a week: from {hours:.1f} to {more_hours:.1f}",
                    {**student, "study_hours": more_hours},
                )
            )
        return scenarios

    # ---- warnings
    def _warnings(self, student: dict, codes: List[str]) -> List[str]:
        warnings: List[str] = []
        selected = [self.courses.loc[c] for c in codes]

        given = student.get("category_averages") or {}
        needed = sorted({str(c["category"]) for c in selected} - set(given))
        if needed:
            names = "; ".join(config.CATEGORY_LABELS.get(n, n) for n in needed)
            if given:
                warnings.append(
                    f"No average given for these course areas: {names}. Your overall average was used for those courses."
                )
            else:
                warnings.append(
                    "No averages by course area were given, so your overall average was used. "
                    "Adding them makes predictions more personal."
                )

        if student.get("prereq_avg_score") is None and any(int(c["has_prereqs"]) for c in selected):
            warnings.append(
                "No prerequisite average was given, so your overall average was used instead."
            )

        for key, (low, high, label) in config.TRAINING_RANGES.items():
            value = float(student[key])
            if value < low or value > high:
                shown = f"{value * 100:.0f}%" if key == "attendance" else f"{value:g}"
                shown_range = (
                    f"{low * 100:.0f}% to {high * 100:.0f}%"
                    if key == "attendance"
                    else f"{low:g} to {high:g}"
                )
                warnings.append(
                    f"{label} of {shown} is outside the range the model learned from ({shown_range}). "
                    "Treat these predictions with extra caution."
                )
        return warnings

    # ---- main entry point
    def recommend(self, student: dict, course_codes: List[str]) -> dict:
        codes = list(dict.fromkeys(course_codes))  # drop duplicates, keep order
        unknown = [c for c in codes if c not in self.courses.index]
        if unknown:
            raise UnknownCourseError(unknown)

        base = self.predict_scores(student, codes)
        scenarios = self._scenarios(student)
        scenario_scores = [self.predict_scores(variant, codes) for _, variant in scenarios]

        predictions = []
        for i, code in enumerate(codes):
            course = self._course_dict(self.courses.loc[code])
            score = float(base[i])
            letter, _ = grade_for(score)
            band = band_for(score)
            borderline = _is_borderline(score)

            what_if = []
            for (action, _), scores in zip(scenarios, scenario_scores):
                new_score = float(scores[i])
                what_if.append(
                    {
                        "action": action,
                        "predicted_score": round(new_score, 1),
                        "change": round(new_score - score, 1),
                    }
                )

            predictions.append(
                {
                    **{k: course[k] for k in ("course_code", "category", "level", "semester", "units", "is_elective")},
                    "predicted_score": round(score, 1),
                    "predicted_grade": letter,
                    "band": band,
                    "recommendation": _RECOMMENDATION[band],
                    "borderline": borderline,
                    "advice": _advice(band, borderline),
                    "what_if": what_if,
                }
            )

        predictions.sort(key=lambda p: p["predicted_score"], reverse=True)

        return {
            "predictions": predictions,
            "summary": self._summary(predictions, base, codes),
            "warnings": self._warnings(student, codes),
            "model": self.model_info(),
        }

    def _summary(self, predictions: List[dict], base: np.ndarray, codes: List[str]) -> dict:
        units = np.array([self.courses.loc[c, "units"] for c in codes], dtype=float)
        points = np.array([grade_for(float(s))[1] for s in base], dtype=float)
        total_units = float(units.sum())

        gpa = float((units * points).sum() / total_units)
        average = float((units * base).sum() / total_units)
        at_risk = [p["course_code"] for p in predictions if p["band"] == "At Risk"]

        if not at_risk:
            message = "None of these courses are predicted to be high risk."
        elif len(at_risk) == 1:
            message = f"One course is predicted to be high risk: {at_risk[0]}."
        else:
            message = (
                f"{len(at_risk)} courses are predicted to be high risk: {', '.join(at_risk)}. "
                "Consider spreading them across semesters if your department allows it."
            )

        return {
            "course_count": len(codes),
            "total_units": int(total_units),
            "predicted_gpa": round(gpa, 2),
            "expected_average_score": round(average, 1),
            "high_risk_count": len(at_risk),
            "message": message,
        }
