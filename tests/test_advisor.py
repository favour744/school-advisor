"""Tests for the advisor logic and request validation. No web server needed.

Run:  python -m unittest discover -s tests -v     (or: pytest)
These need model/advisor_model.pkl and model/course_info.csv; they are skipped if missing.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pydantic import ValidationError

from app.advisor import Advisor, ModelLoadError, UnknownCourseError, band_for, grade_for
from app.schemas import RecommendRequest, StudentProfile

STUDENT = {
    "jamb_score": 230,
    "study_hours": 11.0,
    "attendance": 0.82,
    "prior_cgpa": 3.7,
    "prior_semester_gpa": 3.6,
    "prior_avg_score": 64.0,
    "courses_failed_so_far": 0,
    "courses_taken_so_far": 15,
    "category_averages": {"Programming": 66.0, "AI_Data": 61.0},
}


class AdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.advisor = Advisor.load()
        except ModelLoadError as exc:
            raise unittest.SkipTest(str(exc))
        cls.slot = cls.advisor.slot_codes(200, 1, include_electives=False)

    # ---- data and loading
    def test_course_table(self):
        courses = self.advisor.course_list()
        self.assertEqual(len(courses), 47)
        self.assertEqual(len({c["course_code"] for c in courses}), 47)
        self.assertTrue(all(c["units"] in (2, 3) for c in courses))

    def test_course_filter(self):
        rows = self.advisor.course_list(level=100, semester=2)
        self.assertTrue(rows)
        self.assertTrue(all(c["level"] == 100 and c["semester"] == 2 for c in rows))
        self.assertIn("COS 102", [c["course_code"] for c in rows])

    def test_slot_codes_electives(self):
        without = self.advisor.slot_codes(100, 2, include_electives=False)
        with_electives = self.advisor.slot_codes(100, 2, include_electives=True)
        self.assertLess(len(without), len(with_electives))
        self.assertTrue(set(without) <= set(with_electives))

    def test_missing_files_give_clear_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ModelLoadError) as ctx:
                Advisor.load(Path(tmp) / "nope.pkl", Path(tmp) / "nope.csv", Path(tmp) / "m.json")
        self.assertIn("Model file not found", str(ctx.exception))

    def test_corrupt_model_file_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.pkl"
            bad.write_bytes(b"not a pickle")
            with self.assertRaises(ModelLoadError) as ctx:
                Advisor.load(bad, Path(tmp) / "course_info.csv", Path(tmp) / "m.json")
        # either the model or the course table is reported first; both are fine
        self.assertTrue(
            "Could not load" in str(ctx.exception) or "Course table not found" in str(ctx.exception)
        )

    # ---- predictions
    def test_matches_the_pipeline_called_directly(self):
        """The app must feed the model exactly what the notebook would."""
        code = "CSC 301"
        course = self.advisor.courses.loc[code]
        row = pd.DataFrame(
            [
                {
                    "course_code": code,
                    "jamb_score": 230,
                    "level": int(course["level"]),
                    "semester": int(course["semester"]),
                    "units": int(course["units"]),
                    "is_elective": int(course["is_elective"]),
                    "category": course["category"],
                    "study_hours": 11.0,
                    "attendance": 0.82,
                    "prior_cgpa": 3.7,
                    "prior_semester_gpa": 3.6,
                    "prior_avg_score": 64.0,
                    "courses_failed_so_far": 0,
                    "courses_taken_so_far": 15,
                    "category_avg_prior": 66.0,  # Programming average given
                    "has_category_history": 1,
                    "prereq_avg_score": 64.0,  # falls back to the overall average
                    "has_prereqs": int(course["has_prereqs"]),
                }
            ]
        )
        expected = float(self.advisor.model.predict(row[self.advisor.features])[0])
        result = self.advisor.recommend(STUDENT, [code])
        self.assertAlmostEqual(result["predictions"][0]["predicted_score"], round(expected, 1), places=1)

    def test_results_are_ranked_best_first(self):
        result = self.advisor.recommend(STUDENT, self.slot)
        scores = [p["predicted_score"] for p in result["predictions"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(scores), len(self.slot))

    def test_duplicate_codes_are_counted_once(self):
        result = self.advisor.recommend(STUDENT, ["CSC 301", "CSC 301", "CSC 309"])
        self.assertEqual(len(result["predictions"]), 2)

    def test_unknown_course_is_reported(self):
        with self.assertRaises(UnknownCourseError) as ctx:
            self.advisor.recommend(STUDENT, ["CSC 301", "XYZ 999"])
        self.assertEqual(ctx.exception.codes, ["XYZ 999"])

    def test_stronger_record_predicts_higher(self):
        weak = {**STUDENT, "prior_avg_score": 45.0, "prior_cgpa": 2.0, "prior_semester_gpa": 2.0,
                "category_averages": {"Programming": 42.0}}
        strong = {**STUDENT, "prior_avg_score": 80.0, "prior_cgpa": 4.6, "prior_semester_gpa": 4.7,
                  "category_averages": {"Programming": 82.0}}
        low = self.advisor.recommend(weak, ["CSC 301"])["predictions"][0]["predicted_score"]
        high = self.advisor.recommend(strong, ["CSC 301"])["predictions"][0]["predicted_score"]
        self.assertGreater(high, low + 10)

    def test_scores_stay_between_0_and_100(self):
        best = {**STUDENT, "prior_avg_score": 100.0, "prior_cgpa": 5.0, "prior_semester_gpa": 5.0,
                "jamb_score": 400, "attendance": 1.0, "study_hours": 40.0,
                "category_averages": {"Professional": 100.0}}
        worst = {**STUDENT, "prior_avg_score": 0.0, "prior_cgpa": 0.0, "prior_semester_gpa": 0.0,
                 "jamb_score": 100, "attendance": 0.0, "study_hours": 0.0,
                 "courses_failed_so_far": 15, "category_averages": {"Professional": 0.0}}
        for student in (best, worst):
            for p in self.advisor.recommend(student, self.slot)["predictions"]:
                self.assertGreaterEqual(p["predicted_score"], 0)
                self.assertLessEqual(p["predicted_score"], 100)

    # ---- grades, bands, summary
    def test_grade_cutoffs(self):
        cases = {100: "A", 70: "A", 69.9: "B", 60: "B", 59.9: "C", 50: "C", 49.9: "D",
                 45: "D", 44.9: "E", 40: "E", 39.9: "F", 0: "F"}
        for score, letter in cases.items():
            self.assertEqual(grade_for(score)[0], letter, score)

    def test_band_cutoffs(self):
        self.assertEqual(band_for(49.9), "At Risk")
        self.assertEqual(band_for(50), "Average")
        self.assertEqual(band_for(64.9), "Average")
        self.assertEqual(band_for(65), "Strong")

    def test_summary_gpa_matches_the_courses(self):
        result = self.advisor.recommend(STUDENT, self.slot)
        points = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
        units = {c: int(self.advisor.courses.loc[c, "units"]) for c in self.slot}
        total = sum(units.values())
        # Grades are shown for the rounded score; recompute from the unrounded path
        # by tolerating a small gap rather than an exact match.
        weighted = sum(units[p["course_code"]] * points[p["predicted_grade"]] for p in result["predictions"])
        self.assertAlmostEqual(result["summary"]["predicted_gpa"], weighted / total, delta=0.2)
        self.assertEqual(result["summary"]["total_units"], total)
        self.assertEqual(result["summary"]["course_count"], len(self.slot))
        at_risk = sum(p["band"] == "At Risk" for p in result["predictions"])
        self.assertEqual(result["summary"]["high_risk_count"], at_risk)

    def test_what_if_changes_are_consistent(self):
        result = self.advisor.recommend({**STUDENT, "attendance": 0.6, "study_hours": 5.0}, self.slot)
        for p in result["predictions"]:
            self.assertEqual(len(p["what_if"]), 2)
            for w in p["what_if"]:
                self.assertAlmostEqual(w["change"], w["predicted_score"] - p["predicted_score"], delta=0.11)

    def test_no_attendance_idea_when_already_perfect(self):
        result = self.advisor.recommend({**STUDENT, "attendance": 1.0}, ["CSC 301"])
        actions = [w["action"] for w in result["predictions"][0]["what_if"]]
        self.assertFalse(any("Attend" in a for a in actions))

    # ---- warnings
    def test_warns_when_optional_detail_is_missing(self):
        bare = {k: v for k, v in STUDENT.items() if k != "category_averages"}
        warnings = self.advisor.recommend(bare, ["CSC 301"])["warnings"]
        self.assertTrue(any("averages by course area" in w for w in warnings))
        self.assertTrue(any("prerequisite average" in w for w in warnings))

    def test_warns_about_inputs_outside_training_range(self):
        warnings = self.advisor.recommend({**STUDENT, "jamb_score": 395}, ["CSC 301"])["warnings"]
        self.assertTrue(any("JAMB score" in w for w in warnings))


class SchemaTests(unittest.TestCase):
    def good(self, **changes):
        data = {k: v for k, v in STUDENT.items()}
        data.update(changes)
        return data

    def test_valid_profile(self):
        self.assertEqual(StudentProfile(**self.good()).jamb_score, 230)

    def test_rejects_out_of_range_values(self):
        for field, value in [("attendance", 1.5), ("prior_cgpa", 5.5), ("jamb_score", 50),
                             ("study_hours", -1), ("prior_avg_score", 101)]:
            with self.assertRaises(ValidationError, msg=field):
                StudentProfile(**self.good(**{field: value}))

    def test_failed_cannot_exceed_taken(self):
        with self.assertRaises(ValidationError) as ctx:
            StudentProfile(**self.good(courses_failed_so_far=20, courses_taken_so_far=10))
        self.assertIn("Courses failed cannot be more than courses taken", str(ctx.exception))

    def test_unknown_fields_and_categories_are_rejected(self):
        with self.assertRaises(ValidationError):
            StudentProfile(**self.good(favourite_colour="blue"))
        with self.assertRaises(ValidationError):
            StudentProfile(**self.good(category_averages={"Cooking": 70}))

    def test_recommend_needs_codes_or_slot(self):
        with self.assertRaises(ValidationError):
            RecommendRequest(student=self.good())
        with self.assertRaises(ValidationError):
            RecommendRequest(student=self.good(), level=200)
        self.assertIsNotNone(RecommendRequest(student=self.good(), level=200, semester=1))
        self.assertIsNotNone(RecommendRequest(student=self.good(), course_codes=["CSC 301"]))


if __name__ == "__main__":
    unittest.main()
