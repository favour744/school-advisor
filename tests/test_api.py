"""HTTP tests for the FastAPI app, using FastAPI's TestClient.

Run:  python -m unittest discover -s tests -v     (or: pytest)
Needs the dev requirements (httpx) and model/advisor_model.pkl; skipped otherwise.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi.testclient import TestClient

    from app.advisor import Advisor, ModelLoadError
    from app.main import app

    Advisor.load()
    SKIP_REASON = None
except ImportError as exc:  # fastapi or httpx not installed
    SKIP_REASON = f"FastAPI test dependencies missing: {exc}"
except ModelLoadError as exc:
    SKIP_REASON = str(exc)

STUDENT = {
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


@unittest.skipIf(SKIP_REASON, SKIP_REASON or "")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_manager = TestClient(app)
        cls.client = cls.client_manager.__enter__()  # runs the lifespan, which loads the model

    @classmethod
    def tearDownClass(cls):
        cls.client_manager.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok", "model_loaded": True, "detail": None})

    def test_courses(self):
        r = self.client.get("/api/courses")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 47)

    def test_courses_filtered(self):
        r = self.client.get("/api/courses", params={"level": 100, "semester": 2})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(all(c["level"] == 100 and c["semester"] == 2 for c in r.json()))

    def test_courses_bad_filter(self):
        self.assertEqual(self.client.get("/api/courses", params={"level": 150}).status_code, 422)
        self.assertEqual(self.client.get("/api/courses", params={"semester": 3}).status_code, 422)

    def test_predict_one_course(self):
        r = self.client.post("/api/predict", json={"student": STUDENT, "course_code": "CSC 301"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["prediction"]["course_code"], "CSC 301")
        self.assertTrue(0 <= body["prediction"]["predicted_score"] <= 100)
        self.assertIn(body["prediction"]["band"], ("At Risk", "Average", "Strong"))
        self.assertTrue(body["model"]["name"])

    def test_recommend_with_codes(self):
        codes = ["CSC 301", "CSC 309", "VUA-CSC 303"]
        r = self.client.post("/api/recommend", json={"student": STUDENT, "course_codes": codes})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        scores = [p["predicted_score"] for p in body["predictions"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(body["summary"]["course_count"], 3)

    def test_recommend_by_level_and_semester(self):
        r = self.client.post("/api/recommend", json={"student": STUDENT, "level": 200, "semester": 1})
        self.assertEqual(r.status_code, 200, r.text)
        codes = {p["course_code"] for p in r.json()["predictions"]}
        self.assertTrue(codes)
        self.assertFalse(any(p["is_elective"] for p in r.json()["predictions"]))

        with_electives = self.client.post(
            "/api/recommend",
            json={"student": STUDENT, "level": 200, "semester": 1, "include_electives": True},
        ).json()
        self.assertGreater(len(with_electives["predictions"]), len(codes))

    def test_unknown_course_is_422_with_message(self):
        r = self.client.post("/api/recommend", json={"student": STUDENT, "course_codes": ["XYZ 999"]})
        self.assertEqual(r.status_code, 422)
        self.assertIn("XYZ 999", r.json()["detail"])

    def test_validation_errors_name_the_field(self):
        r = self.client.post(
            "/api/predict", json={"student": {**STUDENT, "attendance": 1.5}, "course_code": "CSC 301"}
        )
        self.assertEqual(r.status_code, 422)
        self.assertIn("attendance", str(r.json()["detail"]))

    def test_failed_more_than_taken_is_rejected(self):
        bad = {**STUDENT, "courses_failed_so_far": 30, "courses_taken_so_far": 10}
        r = self.client.post("/api/predict", json={"student": bad, "course_code": "CSC 301"})
        self.assertEqual(r.status_code, 422)

    def test_recommend_needs_codes_or_slot(self):
        r = self.client.post("/api/recommend", json={"student": STUDENT})
        self.assertEqual(r.status_code, 422)

    def test_503_when_model_is_not_loaded(self):
        advisor, error = app.state.advisor, app.state.load_error
        app.state.advisor, app.state.load_error = None, "Model file not found: test"
        try:
            health = self.client.get("/api/health").json()
            self.assertEqual(health["status"], "degraded")
            self.assertFalse(health["model_loaded"])
            r = self.client.get("/api/courses")
            self.assertEqual(r.status_code, 503)
            self.assertIn("Model file not found", r.json()["detail"])
        finally:
            app.state.advisor, app.state.load_error = advisor, error

    def test_web_page_and_assets_are_served(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Course advisor", page.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/styles.css").status_code, 200)

    def test_openapi_docs_available(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        self.assertIn("/api/recommend", self.client.get("/openapi.json").json()["paths"])


if __name__ == "__main__":
    unittest.main()
