# Academic advisor

A FastAPI app that predicts how a student is likely to score in each course, ranks the
courses, and turns the predictions into advice. The model is a scikit-learn pipeline
(Ridge regression) trained in Google Colab on `data/model_ready.csv` and saved as a `.pkl` file.

- Web page at `/` : enter your record, pick courses, see predicted score, grade and band for each
- JSON API under `/api` with interactive docs at `/docs`

## Project layout

```
academic-advisor/
├── app/
│   ├── main.py        FastAPI routes and startup
│   ├── advisor.py     loads the model, builds model inputs, predicts, writes advice
│   ├── schemas.py     request/response models and input validation
│   ├── config.py      paths, grade scale, band cut-offs
│   └── static/        the web page (index.html, styles.css, app.js)
├── model/
│   ├── advisor_model.pkl   <- from your Colab notebook (see step 3)
│   ├── course_info.csv     one row per course; used to look up category, units, level
│   └── metrics.json        test scores shown on the page
├── data/model_ready.csv    the training data
├── tests/                  unit tests for the logic and the HTTP API
├── train_model.py          rebuilds the model locally with the same recipe as Colab
├── requirements.txt
└── requirements-dev.txt
```

## Run it locally

You need Python 3.10 to 3.13 and Git.

**1. Create a virtual environment and install packages**

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

**2. Check the scikit-learn version**

`requirements.txt` pins `scikit-learn==1.6.1`, the version Colab printed when the model was saved.
A `.pkl` file must be loaded with the same version that saved it.

**3. Put the model in `model/`**

Either copy `advisor_model.pkl` (downloaded from Colab) into `model/`, or build one here:

```bash
python train_model.py
```

`course_info.csv` and `metrics.json` are already included. `train_model.py` regenerates them too.

**4. Start the server**

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> for the web page and <http://127.0.0.1:8000/docs> for the API docs.

If the model is missing or will not load, the page shows the reason in a red notice and
`/api/health` reports `"status": "degraded"`.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health` | Service status and whether the model loaded |
| GET | `/api/courses?level=200&semester=1` | Courses the model knows (filters optional) |
| POST | `/api/predict` | Prediction and advice for one course |
| POST | `/api/recommend` | Predictions for several courses, best first, plus a summary |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "student": {
      "jamb_score": 230, "study_hours": 11, "attendance": 0.82,
      "prior_cgpa": 3.7, "prior_semester_gpa": 3.6, "prior_avg_score": 64,
      "courses_failed_so_far": 0, "courses_taken_so_far": 15,
      "category_averages": {"Programming": 66, "AI_Data": 61}
    },
    "level": 200, "semester": 1
  }'
```

Send either `course_codes` (a list) or both `level` and `semester`. `attendance` is a fraction
from 0 to 1. `category_averages` and `prereq_avg_score` are optional; anything missing falls back
to `prior_avg_score`, which is how the training data treats missing history.

## How the advice is made

1. For each course, the app combines the student's record with the course's own details
   (category, units, level, semester, whether it has prerequisites) into one row.
2. The pipeline predicts a score from 0 to 100.
3. The score becomes a letter grade (A from 70, B from 60, C from 50, D from 45, E from 40) and a band:
   At Risk below 50, Average from 50 to 64, Strong from 65.
4. "What could help" re-runs the prediction with 10 points more attendance and 3 more study hours a week.
5. The summary turns grades into a predicted GPA on the 5-point scale, weighted by units.

## Tests

```bash
python -m unittest discover -s tests -v
```

`tests/test_advisor.py` covers the prediction logic and input validation (it needs the model file).
`tests/test_api.py` covers every endpoint and the 503 behaviour (it needs `requirements-dev.txt`).
Tests that cannot run are skipped with the reason shown.

## Things to know

- **Accuracy.** On held-out students the model is typically off by about 6.6 points (R² 0.645).
  Treat predictions as estimates. The page shows this error as a pale band on each bar.
- **Who it was trained on.** The data starts at 100 level, second semester, so first-semester
  freshmen are outside what the model has seen. Inputs outside the training range trigger a warning.
- **The "what could help" numbers** come from a linear model, so each nudge changes every course by
  about the same amount. They describe associations in past records, not guaranteed effects.
- **Leakage.** `grade` and `grade_band` are computed from `score`, so they are never used as inputs.
  Students are split into train and test by `student_id` so no student appears in both.
- **Pickle files can run code when loaded.** Only load `.pkl` files you made or trust.

## Git and GitHub

```bash
git init
git add .
git commit -m "Academic advisor: model, FastAPI app and tests"
git branch -M main
git remote add origin https://github.com/<your-username>/academic-advisor.git
git push -u origin main
```

## Deploying

Any host that runs Python web apps works (Render, Railway, Fly.io). Use these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Commit `model/advisor_model.pkl` so the host has it.
