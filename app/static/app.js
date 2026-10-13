"use strict";

/* Course advisor front end. Talks to the JSON API under /api.
   All text from the server goes into the page with textContent (via el()),
   never innerHTML, so nothing it sends can inject markup. */

// Must match AT_RISK_BELOW and STRONG_FROM in app/config.py.
const AT_RISK_BELOW = 50;
const STRONG_FROM = 65;

const STORAGE_KEY = "course-advisor-form-v1";

const FIELDS = [
  { name: "jamb_score", label: "JAMB score", hint: "Your UTME score, from 100 to 400.", min: 100, max: 400, value: 220, integer: true },
  { name: "study_hours", label: "Study hours per week", hint: "Time spent studying outside class.", min: 0, max: 60, value: 10 },
  { name: "attendance", label: "Class attendance (%)", hint: "The share of classes you attend.", min: 0, max: 100, value: 77, toApi: (v) => v / 100 },
  { name: "prior_cgpa", label: "Current CGPA", hint: "On the 5-point scale.", min: 0, max: 5, value: 3.6 },
  { name: "prior_semester_gpa", label: "Last semester's GPA", hint: "On the 5-point scale.", min: 0, max: 5, value: 3.5 },
  { name: "prior_avg_score", label: "Average score so far", hint: "The mean of all your course scores, out of 100.", min: 0, max: 100, value: 63 },
  { name: "courses_taken_so_far", label: "Courses taken so far", hint: "Include courses you failed.", min: 0, max: 100, value: 15, integer: true },
  { name: "courses_failed_so_far", label: "Courses failed so far", hint: "Courses you still owe or retook.", min: 0, max: 100, value: 0, integer: true },
];

const CATEGORY_LABELS = {
  AI_Data: "AI and data",
  Software_Apps: "Software and apps",
  Systems_Hardware: "Systems and hardware",
  Programming: "Programming",
  Professional: "Professional",
  Theory: "Theory",
  Networks_Cloud_Security: "Networks, cloud and security",
};

const FIELD_LABELS = Object.fromEntries(FIELDS.map((f) => [f.name, f.label]));
FIELD_LABELS.prereq_avg_score = "Prerequisite average";

const state = { courses: [] };

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ helpers */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child);
  }
  return node;
}

function describeIssue(issue) {
  const field = Array.isArray(issue.loc) ? issue.loc[issue.loc.length - 1] : null;
  const label = FIELD_LABELS[field];
  const message = String(issue.msg || "Invalid value")
    .replace(/^Value error, /, "")
    .replace(/\.$/, "");
  return label ? `${label}: ${message}.` : `${message}.`;
}

function describeError(status, data) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail.map(describeIssue).join(" ");
  return `Something went wrong (error ${status}). Try again.`;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (err) {
    throw new Error("Can't reach the advisor service. Check that the server is running, then try again.");
  }
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    /* the body was not JSON; describeError copes with null */
  }
  if (!response.ok) throw new Error(describeError(response.status, data));
  return data;
}

const bandClass = (band) => `band-${band.toLowerCase().replace(/\s+/g, "-")}`;
const signed = (n) => `${n > 0 ? "+" : ""}${n.toFixed(1)}`;
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/* ---------------------------------------------------------- saved form values */
function readSaved() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch (err) {
    return {};
  }
}

function saveForm() {
  const values = {};
  document.querySelectorAll("#record-fields input, #optional-fields input").forEach((input) => {
    values[input.id] = input.value;
  });
  values.slot = $("slot").value;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(values));
  } catch (err) {
    /* storage can be blocked; the form still works without it */
  }
}

/* ------------------------------------------------------------------- the form */
function numberField(id, def) {
  return el(
    "div",
    { class: "field" },
    el("label", { for: id, text: def.label }),
    el("input", {
      id,
      type: "number",
      inputmode: def.integer ? "numeric" : "decimal",
      min: def.min,
      max: def.max,
      step: "any",
      required: def.required,
      value: def.value,
      placeholder: def.placeholder,
      "aria-describedby": def.hint ? `${id}-hint` : null,
    }),
    def.hint ? el("p", { class: "hint", id: `${id}-hint`, text: def.hint }) : null
  );
}

function buildFields() {
  $("record-fields").replaceChildren(
    ...FIELDS.map((def) => numberField(`f-${def.name}`, { ...def, required: true }))
  );

  const optional = Object.entries(CATEGORY_LABELS).map(([key, label]) =>
    numberField(`o-${key}`, { label: `${label} average`, min: 0, max: 100, placeholder: "Optional" })
  );
  optional.push(
    numberField("o-prereq", {
      label: "Prerequisite average",
      hint: "Your average in the prerequisites for these courses.",
      min: 0,
      max: 100,
      placeholder: "Optional",
    })
  );
  $("optional-fields").replaceChildren(...optional);
}

function readStudent() {
  const student = {};
  for (const def of FIELDS) {
    const number = Number($(`f-${def.name}`).value);
    if (def.integer && !Number.isInteger(number)) {
      throw new Error(`${def.label} must be a whole number.`);
    }
    student[def.name] = def.toApi ? def.toApi(number) : number;
  }

  const averages = {};
  for (const key of Object.keys(CATEGORY_LABELS)) {
    const raw = $(`o-${key}`).value.trim();
    if (raw !== "") averages[key] = Number(raw);
  }
  if (Object.keys(averages).length) student.category_averages = averages;

  const prereq = $("o-prereq").value.trim();
  if (prereq !== "") student.prereq_avg_score = Number(prereq);
  return student;
}

/* -------------------------------------------------------------- course picker */
const slotLabel = (level, semester) =>
  `${level} level, ${semester === 1 ? "first" : "second"} semester`;

function buildSlots() {
  const slots = new Map();
  for (const c of state.courses) slots.set(`${c.level}-${c.semester}`, [c.level, c.semester]);
  const sorted = [...slots.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  const select = $("slot");
  select.replaceChildren(
    ...sorted.map(([level, semester]) =>
      el("option", { value: `${level}-${semester}`, text: slotLabel(level, semester) })
    )
  );
  select.disabled = false;
}

function renderCourses() {
  const [level, semester] = $("slot").value.split("-").map(Number);
  const items = state.courses
    .filter((c) => c.level === level && c.semester === semester)
    .sort((a, b) => Number(a.is_elective) - Number(b.is_elective) || a.course_code.localeCompare(b.course_code));

  $("course-list").replaceChildren(
    ...items.map((c) => {
      const area = CATEGORY_LABELS[c.category] || c.category;
      const meta = `${area}, ${plural(c.units, "unit", "units")}${c.is_elective ? ", elective" : ""}`;
      return el(
        "li",
        {},
        el(
          "label",
          { class: "course" },
          el("input", { type: "checkbox", name: "course", value: c.course_code, checked: !c.is_elective }),
          el("span", { class: "course-code", text: c.course_code }),
          el("span", { class: "course-meta", text: meta })
        )
      );
    })
  );
}

/* -------------------------------------------------------------------- results */
function trackNode(prediction, typicalError) {
  const node = el(
    "div",
    {
      class: "track",
      role: "img",
      "aria-label": `Predicted score ${prediction.predicted_score} out of 100, typically off by about ${typicalError} points either way.`,
    },
    el("span", { class: "zone zone-risk" }),
    el("span", { class: "zone zone-average" }),
    el("span", { class: "zone zone-strong" }),
    el("span", { class: "whisker" }),
    el("span", { class: "marker" })
  );
  node.dataset.score = prediction.predicted_score;
  node.dataset.error = typicalError;
  return node;
}

function noteNode(prediction) {
  const helpful = prediction.what_if.filter((w) => w.change >= 0.1);
  const showIdeas = prediction.band !== "Strong" && helpful.length > 0;

  return el(
    "div",
    { class: "note" },
    el("p", { text: prediction.advice }),
    showIdeas
      ? el(
          "p",
          {},
          "What could help: ",
          helpful.map((w, i) => [
            i > 0 ? " " : "",
            `${w.action}, about `,
            el("strong", { text: `${signed(w.change)} points` }),
            ".",
          ])
        )
      : null
  );
}

function rowNode(prediction, typicalError) {
  const area = CATEGORY_LABELS[prediction.category] || prediction.category;
  const meta = `${area}, ${plural(prediction.units, "unit", "units")}${prediction.is_elective ? ", elective" : ""}`;

  return el(
    "li",
    { class: "row" },
    el(
      "div",
      { class: "who" },
      el("span", { class: "who-code", text: prediction.course_code }),
      el("span", { class: "who-meta", text: meta })
    ),
    trackNode(prediction, typicalError),
    el(
      "div",
      { class: "outcome" },
      el("span", { class: "score", text: prediction.predicted_score.toFixed(1) }),
      el("span", {
        class: `outcome-detail ${bandClass(prediction.band)}`,
        text: `Grade ${prediction.predicted_grade}, ${prediction.band}`,
      })
    ),
    noteNode(prediction)
  );
}

function scaleNode() {
  const tick = (value, extra) =>
    el("span", { class: `tick ${extra || ""}`.trim(), style: `left: ${value}%`, text: String(value) });

  const names = el(
    "div",
    { class: "zone-names" },
    el("span", { style: `flex-basis: ${AT_RISK_BELOW}%`, text: "At Risk" }),
    el("span", { style: `flex-basis: ${STRONG_FROM - AT_RISK_BELOW}%`, text: "Average" }),
    el("span", { style: `flex-basis: ${100 - STRONG_FROM}%`, text: "Strong" })
  );
  const ticks = el(
    "div",
    { class: "ticks", "aria-hidden": "true" },
    tick(0, "tick-start"),
    tick(AT_RISK_BELOW),
    tick(STRONG_FROM),
    tick(100, "tick-end")
  );
  return el("div", { class: "scale", "aria-hidden": "true" }, el("div", { class: "scale-track" }, names, ticks));
}

function renderResults(result) {
  const { summary, warnings, predictions, model } = result;
  const error = model.typical_error;

  const headline = el(
    "p",
    { class: "headline" },
    "If this semester goes as predicted, your GPA would be ",
    el("strong", { text: summary.predicted_gpa.toFixed(2) }),
    ` across ${plural(summary.total_units, "unit", "units")}, with an average score of `,
    el("strong", { text: summary.expected_average_score.toFixed(1) }),
    "."
  );

  const notes = warnings.length
    ? el("ul", { class: "notes", "aria-label": "Notes about these predictions" }, warnings.map((w) => el("li", { text: w })))
    : null;

  const modelName = model.r2 === null ? model.name : `${model.name}, R² ${model.r2}`;
  const fineprint = el("p", {
    class: "fineprint",
    text:
      `These are estimates from patterns in past student records (${modelName}). A prediction is typically ` +
      `off by about ${error} points either way, shown as the pale band on each bar. The suggestions describe ` +
      "what went with better results in that data, not a guaranteed effect.",
  });

  $("results").replaceChildren(
    el("h2", { id: "results-heading", tabindex: "-1", text: "Your advice" }),
    headline,
    el("p", { class: "summary-message", text: summary.message }),
    notes,
    scaleNode(),
    el("ol", { class: "rows", "aria-label": "Courses from best to worst predicted score" }, predictions.map((p) => rowNode(p, error))),
    fineprint
  );

  // Move the markers into place on the next frame so they slide from zero.
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      document.querySelectorAll("#results .track").forEach((track) => {
        const score = Number(track.dataset.score);
        const spread = Number(track.dataset.error);
        const low = Math.max(0, score - spread);
        const high = Math.min(100, score + spread);
        track.querySelector(".marker").style.left = `${score}%`;
        const whisker = track.querySelector(".whisker");
        whisker.style.left = `${low}%`;
        whisker.style.width = `${high - low}%`;
      });
    })
  );

  const heading = $("results-heading");
  heading.focus({ preventScroll: true });
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  heading.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
}

/* ------------------------------------------------------------------ messages */
function showNotice(text) {
  const notice = $("service-notice");
  notice.textContent = text;
  notice.hidden = false;
}

function showFormError(text) {
  const box = $("form-error");
  box.textContent = text;
  box.hidden = !text;
}

function setBusy(busy) {
  const button = $("submit");
  button.disabled = busy;
  button.textContent = busy ? "Working out your advice" : "Get advice";
  $("results").setAttribute("aria-busy", busy ? "true" : "false");
}

/* --------------------------------------------------------------------- start */
async function handleSubmit(event) {
  event.preventDefault();
  showFormError("");

  let student;
  try {
    student = readStudent();
  } catch (err) {
    showFormError(err.message);
    return;
  }

  const codes = [...document.querySelectorAll('input[name="course"]:checked')].map((box) => box.value);
  if (codes.length === 0) {
    showFormError("Choose at least one course.");
    return;
  }

  setBusy(true);
  try {
    const result = await api("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ student, course_codes: codes }),
    });
    saveForm();
    renderResults(result);
  } catch (err) {
    showFormError(err.message);
  } finally {
    setBusy(false);
  }
}

async function init() {
  buildFields();

  try {
    const health = await api("/api/health");
    if (!health.model_loaded) {
      showNotice(`The advisor model is not loaded. ${health.detail || ""}`.trim());
      return;
    }
    state.courses = await api("/api/courses");
  } catch (err) {
    showNotice(err.message);
    return;
  }

  buildSlots();

  const saved = readSaved();
  for (const [id, value] of Object.entries(saved)) {
    const input = id === "slot" ? $("slot") : document.getElementById(id);
    if (!input) continue;
    if (id === "slot" && ![...input.options].some((o) => o.value === value)) continue;
    input.value = value;
  }

  renderCourses();
  $("slot").addEventListener("change", renderCourses);
  $("advisor-form").addEventListener("submit", handleSubmit);
  $("submit").disabled = false;
}

init();
