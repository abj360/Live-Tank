# Live Tank Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace this repo’s AquaGuard forecast app with a standalone Live Tank page at `/` (navy/cyan dashboard, live-feed placeholder, dummy stats) and delete forecast-only files.

**Architecture:** One self-contained `live_tracking.html` (HTML/CSS/JS). Flask `app.py` only serves that file and `assets/`. No predict APIs, no LSTM, no new packages. Visual structure follows the current dashboard; tokens are navy `#0A1128`, cyan `#4FC3F7`, muted `#8BA3C7`, panels `#0F1A36`.

**Tech Stack:** Flask, flask-cors, gunicorn, vanilla HTML/CSS/JS, stdlib `unittest`.

## Global Constraints

- No new Python or JS dependencies. Do not add Chart.js. Ask before installing anything new.
- Do not name, link to, or depend on AquaGuard in the shipped UI.
- Do not reuse the UAPB/AquaGuard logo on the page.
- Dummy stats: fish count `12`, confidence `94%`, last updated `Sep 15, 2026 · 12:00`.
- Feed hook: `<img id="tankFeed">` with `src` unset; overlay “Camera stream not connected”.
- JS limited to mobile menu toggle; smooth-scroll via `html { scroll-behavior: smooth }` and `href="#live-feed"`.
- Do not commit unless the user asks.
- Leave `Procfile` and `render.yaml` service name `aquaguard` unchanged.

## File structure

- Create: `live_tracking.html` — the only user-facing page
- Create: `tests/test_app.py` — Flask test-client checks for Live Tank content
- Modify: `app.py` — static server only
- Modify: `requirements.txt` — flask, flask-cors, gunicorn
- Modify: `.gitignore` — drop model/dataset exceptions
- Delete: all AquaGuard forecast/LSTM files listed in Task 3
- Keep: `Procfile`, `render.yaml`, `assets/`, `docs/`

---

### Task 1: Failing Flask page tests

**Files:**
- Create: `tests/test_app.py`
- Delete after this task if still present: `tests/test_fallback.py`, `tests/test_lstm_forecast.py` (old LSTM tests; they must not remain)

**Interfaces:**
- Consumes: Flask `app` from `app.py` (`app.test_client()`)
- Produces: `LiveTankPageTests` with `test_index_ok_live_tank`, `test_index_has_feed_and_stats`, `test_index_omits_aquaguard`, `test_predict_routes_gone`

- [ ] **Step 1: Write the failing tests**

Replace LSTM tests with:

```python
import unittest

from app import app


class LiveTankPageTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_ok_live_tank(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Live Tank", html)
        self.assertIn("View Live Feed", html)
        self.assertIn('id="live-feed"', html)

    def test_index_has_feed_and_stats(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("Live Fish Tank Feed", html)
        self.assertIn('id="tankFeed"', html)
        self.assertIn("Camera stream not connected", html)
        self.assertIn("12", html)
        self.assertIn("94%", html)
        self.assertIn("Sep 15, 2026 · 12:00", html)
        self.assertIn('id="howto"', html)
        self.assertIn('id="about"', html)

    def test_index_omits_aquaguard(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("AquaGuard", html)
        self.assertNotIn("aquaguard", html.lower())

    def test_predict_routes_gone(self):
        self.assertEqual(self.client.get("/api/predict_latest").status_code, 404)
        self.assertEqual(self.client.post("/api/predict_manual").status_code, 404)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_app -v`

Expected: FAIL (import error from LSTM startup, or assertion because `/` still serves AquaGuard).

- [ ] **Step 3: Do not write production code in this task**

---

### Task 2: Static Flask app + Live Tank page

**Files:**
- Create: `live_tracking.html` (complete file below)
- Modify: `app.py` (replace entire file)

**Interfaces:**
- Consumes: nothing from Task 1 except the assertions
- Produces: `app` Flask instance; routes `/` and `/assets/<path:filename>` only; `if __name__ == "__main__"` runner

- [ ] **Step 1: Replace `app.py` with**

```python
from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "live_tracking.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory("assets", filename)


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        use_reloader=False,
    )
```

- [ ] **Step 2: Write `live_tracking.html`**

Create the full page in `live_tracking.html`. Required IDs/copy:

- Title and crest wordmark: Live Tank (CSS pennant, no logo `<img>`)
- Utility + primary nav + mobile menu: Live Feed `#live-feed`, How It Works `#howto`, About `#about`
- Intro with View Live Feed `href="#live-feed"`
- Section `#live-feed`, heading Live Fish Tank Feed, `<img id="tankFeed" alt="Live fish tank camera">` without `src`, overlay Camera stream not connected
- Stats 12 / 94% / Sep 15, 2026 · 12:00
- How It Works three steps (camera, detect, dashboard)
- About blurb with no oxygen/crash/LSTM/AquaGuard
- Footer in-page links only
- Tokens: `#0A1128`, `#4FC3F7`, `#8BA3C7`, `#0F1A36`
- Fonts: Bebas Neue, Archivo Black, Archivo, Open Sans, EB Garamond
- Square panels, 900px stack, 760px hamburger
- JS: `#tbMenuBtn` / `#tbMobileMenu` toggle only (same pattern as the reference title bar)
- `html { scroll-behavior: smooth; }` and `prefers-reduced-motion: reduce` → `scroll-behavior: auto`
- Hide `#tankFeed` when it has no `src` so the browser never shows a broken-image icon

Match the reference title-bar markup (`uapb-titlebar`, `tb-top`, `tb-goldbar`/`tb-nav`, `tb-crest`, `tb-mobile-menu`) with cyan replacing gold.

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m unittest tests.test_app -v`

Expected: `OK` — 4 tests pass.

---

### Task 3: Remove AquaGuard-only files and trim config

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Delete: files listed below
- Keep: `Procfile`, `render.yaml`

**Interfaces:**
- Consumes: Task 2 `app.py` / `live_tracking.html`
- Produces: repo that cannot import LSTM modules

- [ ] **Step 1: Replace `requirements.txt` with**

```
flask>=3.0
flask-cors>=4.0
gunicorn>=21.0
```

- [ ] **Step 2: Replace `.gitignore` with**

```
venv/
__pycache__/
*.pyc
.env
*.log
.DS_Store
.vscode/
Thumbs.db
.pytest_cache/
```

- [ ] **Step 3: Delete AquaGuard-only files**

```
aquaguard_dashboard.html
aquaguard_switch_training_data_prompt.md
requirements-lstm.txt
train_lstm.py
lstm_forecast.py
eval_lstm.py
holdout.py
metrics.py
generate_data.py
download_data.py
feature_importance_check.py
plot_period_comparison.py
finetune_chronos.py
test_chronos.py
benchmark_lstm_vs_chronos.py
chronos_bolt_periods_eval.py
lstm_periods_eval.py
pondsdata_model.py
pondsdata_prep.py
train_pondsdata.py
eval_pondsdata.py
eval_multi_period.py
tests/test_fallback.py
tests/test_lstm_forecast.py
deploy_model/
sample_data/
eval_results/
```

- [ ] **Step 4: Re-run unittest**

Run: `python -m unittest tests.test_app -v`

Expected: `OK` — 4 tests pass.

- [ ] **Step 5: Grep the served page for AquaGuard**

Run: `python -c "from app import app; h=app.test_client().get('/').get_data(as_text=True); assert 'AquaGuard' not in h and 'aquaguard' not in h.lower(); print('no aquaguard in page')"`

Expected: `no aquaguard in page`

---

### Task 4: Browser verification

**Files:** none (manual against running server)

- [ ] **Step 1: Start Flask**

Run: `python app.py`

Expected: listening on port 5000.

- [ ] **Step 2: Open `/` and check**

- Header three-tier, Live Tank crest, cyan nav
- Intro + View Live Feed scrolls to `#live-feed`
- Feed placeholder visible, not a broken image
- Stats 12, 94%, Sep 15, 2026 · 12:00
- How It Works and About
- Footer links stay on this page
- ~900px: feed stacks above stats
- ~760px: hamburger opens/closes and nav links work
- No AquaGuard names or links

- [ ] **Step 3: Stop the server**
