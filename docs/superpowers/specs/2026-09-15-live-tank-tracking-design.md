# Live Tank Tracking — Design Spec

Date: 2026-09-15

## Summary

Turn this repository into a standalone **Live Tank** site: a fish-tank monitoring dashboard with a live-feed stage and placeholder detection stats. AquaGuard’s dissolved-oxygen forecast product is not part of this site. AquaGuard’s existing HTML/CSS is a **styling reference only** (header structure, fonts, square panels, spacing, breakpoints). The shipped page must not name, link to, or depend on AquaGuard.

## Product

- **Site name:** Live Tank
- **Purpose:** Show a live camera view of a fish tank and placeholder tracking stats (count, confidence, last updated).
- **Audience:** Same lab/demo context as the current Render app, but a different product.
- **URL:** Flask serves this page at `/` (the only user-facing page).

## Architecture

- One new page: `live_tracking.html` (HTML + CSS + small JS, same self-contained pattern as the current dashboard).
- `app.py` becomes a static server: `/` → `live_tracking.html`, `/assets/<path>` → files in `assets/`.
- No predict APIs, no LSTM, no CSV upload, no database.
- No new Python or JS dependencies. Do not add Chart.js. Ask before installing anything new.
- Existing Render/Procfile shape stays: gunicorn `app:app`, `requirements.txt` trimmed to Flask + flask-cors + gunicorn.

## Visual language

Borrow AquaGuard’s **structure**, not its gold/white forecast theme.

| Token | Value | Use |
| --- | --- | --- |
| Background | `#0A1128` | Page and header dark surfaces |
| Primary / accent | `#4FC3F7` | Headings, nav bar, buttons, borders, active states |
| Body / muted | `#8BA3C7` | Secondary text; not pure white |
| Panel surface | `#0F1A36` | Cards and feed bezel |
| Fonts | Bebas Neue, Archivo Black, Archivo, Open Sans, EB Garamond | Same Google Fonts import as the reference page |
| Corners | Square (`border-radius: 0`) on instrument panels | Match the reference instrument-panel look |
| Breakpoints | 900px (stack feed/stats), 760px (hamburger nav) | Match the reference site |

Header is the same **three-tier** idea: dark utility row, cyan primary nav, CSS text crest/wordmark reading **Live Tank** (do not reuse the UAPB/AquaGuard logo). No Give / Apply / Workday Student Hub.

Nav labels (this page only): **Live Feed**, **How It Works**, **About**. Crest/home scrolls to top of this page. Utility row repeats those in-page links; all hrefs stay on this page (`#live-feed`, `#howto`, `#about`).

## Page sections

1. **Intro** — Eyebrow + display headline + lead paragraph explaining that this dashboard shows a live tank camera and fish-tracking readouts. Copy describes Live Tank only; do not mention AquaGuard, oxygen, crashes, or forecasting. A **View Live Feed** anchor, styled like the reference primary button, smooth-scrolls to `#live-feed`.
2. **Live Fish Tank Feed** (`id="live-feed"`) — Prominent 16:9 video stage. Heading: “Live Fish Tank Feed”. An `<img id="tankFeed">` inside the stage is the hook for a later MJPEG URL; leave `src` unset in this iteration. A “Camera stream not connected” overlay sits on top so an empty `src` never looks like a broken image. Beside the stage (stacked under 900px): three stat cards with dummy data hardcoded in HTML:
   - Live fish count: `12`
   - Detection confidence: `94%`
   - Last updated: `Sep 15, 2026 · 12:00`
3. **How It Works** (`id="howto"`) — Three-step numbered grid in the reference section style: (1) camera on the tank, (2) detect fish in the frame, (3) show count and confidence on this dashboard.
4. **About** (`id="about"`) — Short original blurb for Live Tank. No LSTM, oxygen crash, aerator, or AquaGuard copy.
5. **Footer** — Brand line + in-page links only (Live Feed, How It Works, About). No AquaGuard, AquaVision, or forecast-tool URLs.

## Behavior

- JS is limited to: mobile menu toggle and smooth-scroll to `#live-feed`. No live clock, no scroll-reveal library, no Chart.js.
- Future stream failure: keep the placeholder overlay; do not show a broken media icon.
- No backend for counts or camera in this iteration.

## Repo cleanup

After the new page is in place, remove AquaGuard-only files so this repo matches Live Tank.

**Delete:**

- `aquaguard_dashboard.html`
- `aquaguard_switch_training_data_prompt.md`
- `requirements-lstm.txt`
- `train_lstm.py`, `lstm_forecast.py`, `eval_lstm.py`, `holdout.py`, `metrics.py`, `generate_data.py`, `download_data.py`, `feature_importance_check.py`, `plot_period_comparison.py`
- `finetune_chronos.py`, `test_chronos.py`, `benchmark_lstm_vs_chronos.py`, `chronos_bolt_periods_eval.py`, `lstm_periods_eval.py`
- `pondsdata_model.py`, `pondsdata_prep.py`, `train_pondsdata.py`, `eval_pondsdata.py`, `eval_multi_period.py`
- `tests/` (entire directory)
- `deploy_model/` (entire directory)
- `sample_data/` (entire directory)
- `eval_results/` (entire directory)

**Keep / rewrite:**

- `live_tracking.html` (new)
- `app.py` (static server only)
- `Procfile`
- `render.yaml` (service name may stay `aquaguard` so the existing Render app still deploys)
- `.gitignore` (drop model/dataset exceptions that no longer apply)
- `assets/` if the folder exists (unused logo files may remain; the page must not reference them)
- `requirements.txt` (Flask, flask-cors, gunicorn only)

Do not add links from Live Tank to the old forecast product, GitHub AquaGuard, or the previous dashboard.

## Testing

- Start Flask and open `/`. Confirm header, intro, View Live Feed scroll, feed placeholder, three dummy stats, How It Works, About, footer.
- Resize to 900px and 760px: feed stacks above stats; hamburger nav works.
- Confirm no AquaGuard names or links in the UI.
- No pytest suite after LSTM tests are removed.

## Out of scope

- Wiring a real RTSP/MJPEG stream
- Real fish-detection model or APIs
- New CSS/JS frameworks or packages
- Renaming the GitHub repo or Render service
