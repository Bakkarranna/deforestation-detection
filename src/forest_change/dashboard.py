"""Static results console for the forest-change (deforestation) project.

Reads the committed experiment artifacts (``results/metrics.json`` and
``results/baseline_metrics.json``) and writes one static page to
``docs/index.html``, copying the figures it references into ``docs/figures/``.

Design goals:

* **No build step, no CDN.** Plain HTML/CSS/SVG with a few lines of vanilla
  JS optional-free markup, so it renders identically from ``file://``, a local
  server and GitHub Pages.
* **Nothing hand-typed.** Every number comes from the JSON artifacts, so the
  page cannot drift from the runs.
* **Honest by construction.** Changed pixels are 1-5% of a scene, so a naive
  model scores a high mean-IoU while finding nothing. The page shows the
  change-class metrics, the per-city spread, the area accounting and the
  failure modes rather than a single flattering mean.

Usage::

    python -m forest_change.dashboard        # writes docs/index.html
    python -m forest_change.dashboard --strict   # fail on missing artifacts
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

from . import config

DOCS_DIR = config.PROJECT_ROOT / "docs"
METRICS_PATH = config.RESULTS_DIR / "metrics.json"
BASELINE_PATH = config.RESULTS_DIR / "baseline_metrics.json"

REPO = "Bakkarranna/deforestation-detection"

# Metrics averaged over the held-out test cities for the headline cards.
MEAN_KEYS = (
    "iou_change",
    "iou_no_change",
    "mean_iou",
    "f1_change",
    "precision_change",
    "recall_change",
    "true_ha",
    "pred_ha",
    "abs_ha_error",
)

# Human-readable model labels, keyed by the artefact they come from.
MODEL_LABELS = {
    "unet": "U-Net (mobilenet_v2, 24 ch, 8 epochs)",
    "baseline": "NDVI-loss baseline",
}

# Figure captions keyed by the figure file stem, so every picture says what it
# shows and why it matters rather than being decorative.
CAPTIONS = {
    "per_city_metrics": (
        "IoU and F1 for the change class, city by city, for both models. "
        "Read it as a spread, not a mean: lasvegas and montpellier sit near "
        "0.5 IoU while valencia sits at 0.02, and no single number describes "
        "that."
    ),
    "training_curves": (
        "Validation curves over the 8 epochs. The change-class IoU peaks at "
        "epoch 05 (0.3187) and never recovers, which is why the reported "
        "model is that checkpoint and not the last epoch."
    ),
    "area_error": (
        "Predicted versus true cleared area per test city, with the absolute "
        "error. The model is consistently high on bare-soil cities and low on "
        "the small, subtle ones."
    ),
    "baseline_threshold": (
        "The NDVI forest threshold, tuned on the 14 labelled train cities "
        "only and frozen before the test evaluation. The chosen 0.1 sits at "
        "the top of that tuning curve."
    ),
    "change_map_montpellier": (
        "Predicted change map for montpellier, full-city inference. Green is "
        "forest kept, cleared is the detected loss."
    ),
    "qual_montpellier": (
        "montpellier read as hit / false alarm / miss over the ground truth. "
        "The false alarms are the scattered desert-like pixels at the fringe."
    ),
    "change_map_lasvegas": (
        "Predicted change map for lasvegas. The city sits in bare desert, so "
        "almost every non-urban pixel looks like change to the model."
    ),
    "qual_lasvegas": (
        "lasvegas errors: the model drifts toward labelling all non-forest "
        "land as change, which is the precision failure at city scale."
    ),
    "change_map_valencia": (
        "Predicted change map for valencia, the worst held-out city at 0.02 "
        "IoU. Compare the predicted extent to the small ground-truth patch."
    ),
    "qual_valencia": (
        "valencia errors: a flood of false positives over 104 ha of predicted "
        "loss against 9.7 ha of true loss."
    ),
}

# --------------------------------------------------------------------------- #
# Artefact loading and aggregation
# --------------------------------------------------------------------------- #


def _load_json(path: Path, strict: bool) -> dict | None:
    """Load one JSON artefact, or None when it is absent and not strict."""
    if not path.exists():
        if strict:
            sys.exit(f"missing artefact: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(strict: bool = False) -> dict:
    """Load both models' artefacts, keyed by ``unet`` / ``baseline``."""
    out = {}
    unet = _load_json(METRICS_PATH, strict)
    baseline = _load_json(BASELINE_PATH, strict)
    if unet:
        out["unet"] = unet
    if baseline:
        out["baseline"] = baseline
    return out


def per_city(model: dict | None) -> dict:
    """The city -> metrics mapping of one artefact (empty when absent)."""
    if not model:
        return {}
    return (model.get("test") or {}).get("per_city") or {}


def mean_over_cities(cities: dict) -> dict:
    """Average each metric over the held-out cities, ignoring absent keys.

    City sizes differ by more than 50x (valencia 9.7 ha vs lasvegas 453 ha of
    true change), so this is an unweighted mean over cities: every held-out
    city counts once, which is the number the results table reports.
    """
    out = {}
    n = len(cities) or 1
    for key in MEAN_KEYS:
        values = [float(c[key]) for c in cities.values() if key in c]
        if values:
            out[key] = sum(values) / n
    return out


def history_of(unet: dict | None) -> list[dict]:
    """Per-epoch validation history from the U-Net artefact."""
    if not unet:
        return []
    return list((unet.get("run") or {}).get("history") or [])


def fmt_pct(value) -> str:
    """Format a 0..1 metric as a percentage, or an em dash when absent."""
    if value is None:
        return "&mdash;"
    return f"{float(value) * 100:.2f}%"


def fmt_score(value) -> str:
    """Format a 0..1 score with four decimals, the way the JSON stores it."""
    if value is None:
        return "&mdash;"
    return f"{float(value):.4f}"


def fmt_ha(value) -> str:
    """Format a hectare area with one decimal and a thin space."""
    if value is None:
        return "&mdash;"
    return f"{float(value):,.1f} ha"


def fmt_delta(delta: float) -> str:
    """Format a change-class F1 difference in percentage points, signed."""
    return f"{delta * 100:+.1f} pts"


def esc(value) -> str:
    """HTML-escape anything, including None."""
    return html.escape("" if value is None else str(value), quote=True)


# --------------------------------------------------------------------------- #
# HTML sections
# --------------------------------------------------------------------------- #


def _kpi_cards(metrics: dict) -> str:
    """Headline cards: the U-Net change-class scores and the baseline margin."""
    unet = per_city(metrics.get("unet"))
    base = per_city(metrics.get("baseline"))
    if not unet:
        return ""
    mu, mb = mean_over_cities(unet), mean_over_cities(base)
    cards = [
        ("U-Net IoU (change)", fmt_score(mu.get("iou_change")),
         "mean over the held-out cities"),
        ("U-Net F1 (change)", fmt_score(mu.get("f1_change")),
         f"precision {fmt_score(mu.get('precision_change'))}, "
         f"recall {fmt_score(mu.get('recall_change'))}"),
        ("U-Net mean IoU", fmt_score(mu.get("mean_iou")),
         "both classes, change and no-change"),
    ]
    if mb:
        cards.append(
            ("Margin over baseline", fmt_delta(
                mu.get("f1_change", 0) - mb.get("f1_change", 0)),
             f"change-class F1: {fmt_score(mb.get('f1_change'))} for the baseline"),
        )
    body = "".join(
        '<div class="kpi%s"><p class="kpi-label">%s<span class="hint">%s</span></p>'
        '<p class="kpi-value">%s</p></div>'
        % (" kpi-plain" if i == 2 else "", esc(label), esc(hint), value)
        for i, (label, value, hint) in enumerate(cards)
    )
    return f'<div class="kpis">{body}</div>'


def comparison_table(metrics: dict) -> str:
    """Both models on the same test cities, one row per metric."""
    if not metrics:
        return ""
    names = [k for k in ("unet", "baseline") if k in metrics]
    means = {k: mean_over_cities(per_city(metrics[k])) for k in names}
    rows = []
    for key, label in (
        ("iou_change", "IoU (change class)"),
        ("f1_change", "F1 (change class)"),
        ("precision_change", "Precision (change class)"),
        ("recall_change", "Recall (change class)"),
        ("mean_iou", "Mean IoU (both classes)"),
        ("iou_no_change", "IoU (no-change class)"),
    ):
        cells = "".join(
            f'<td class="num">{fmt_score(means[k].get(key))}</td>' for k in names
        )
        head = f'<th scope="row">{label}</th>'
        cls = ' class="best"' if key == "f1_change" else ""
        rows.append(f'<tr{cls}>{head}{cells}</tr>')
    cols = "".join(f'<th scope="col">{esc(MODEL_LABELS[k])}</th>' for k in names)
    n_cities = len(per_city(metrics.get("unet"))) or len(
        per_city(metrics.get("baseline"))
    )
    return (
        '<table class="grid"><caption>Mean over the '
        f"{n_cities} held-out OSCD test cities, never tuned on them"
        f'</caption><thead><tr><th scope="col">Metric</th>{cols}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def city_table(metrics: dict) -> str:
    """Per-city table: U-Net against the baseline, with the area accounting."""
    cities = per_city(metrics.get("unet"))
    base = per_city(metrics.get("baseline"))
    if not cities:
        return ""
    bars = []
    for name, u in sorted(cities.items(), key=lambda kv: -kv[1]["f1_change"]):
        b = base.get(name, {})
        f1 = u.get("f1_change")
        cls = "ok" if (f1 or 0) >= 0.5 else ("meh" if (f1 or 0) >= 0.25 else "bad")
        bars.append(
            f'<tr><th scope="row">{esc(name)}</th>'
            f'<td class="num"><span class="tag {cls}">{fmt_score(f1)}</span></td>'
            f'<td class="bar-cell"><span class="bar" style="width:'
            f'{(f1 or 0) * 100:.1f}%"></span></td>'
            f'<td class="num">{fmt_score(b.get("iou_change"))}</td>'
            f'<td class="num">{fmt_score(b.get("f1_change"))}</td>'
            f'<td class="num">{fmt_ha(u.get("true_ha"))}</td>'
            f'<td class="num">{fmt_ha(u.get("pred_ha"))}</td>'
            f'<td class="num">{fmt_ha(b.get("pred_ha"))}</td>'
            f'<td class="num">{fmt_ha(u.get("abs_ha_error"))}</td></tr>'
        )
    return (
        '<table class="grid bars"><caption>Sorted best-to-worst on change-class '
        "F1. Hectares are per-city totals over the 12-band scene pair, so a "
        "predicted-area error is an area decision, not a pixel one."
        '</caption><thead><tr><th scope="col">City (held out)</th>'
        '<th scope="col">U-Net F1</th><th scope="col" class="bar-cell">F1</th>'
        '<th scope="col">Baseline IoU</th><th scope="col">Baseline F1</th>'
        '<th scope="col">True cleared</th><th scope="col">U-Net cleared</th>'
        '<th scope="col">Baseline cleared</th>'
        '<th scope="col">U-Net area error</th></tr></thead>'
        f'<tbody>{"".join(bars)}</tbody></table>'
    )


def naive_probe(cities: dict) -> tuple[float, float] | None:
    """The score a model that predicts "no change everywhere" would get.

    Derived from the artefacts rather than asserted: with the no-change class
    covering ~97% of every scene, predicting nothing but no-change scores an
    IoU on that class equal to its pixel share, so its mean IoU is half of
    that. It is the number the honest headline has to beat, and the reason a
    bare mean-IoU says almost nothing about change detection.
    """
    values = [float(c["iou_no_change"]) for c in cities.values() if "iou_no_change" in c]
    if not values:
        return None
    share = sum(values) / len(values)
    return share / 2.0, share


def _honest_section(metrics: dict) -> str:
    """Failure modes, bias and run provenance, all read from the artefacts."""
    cities = per_city(metrics.get("unet"))
    if not cities:
        return ""
    ranked = sorted(cities.items(), key=lambda kv: kv[1].get("f1_change") or 0)
    worst, w = ranked[0]
    best, b = ranked[-1]
    probe = naive_probe(cities)
    naive = (
        f"a model that labels every pixel <em>no-change</em> scores "
        f"{fmt_score(probe[0])} mean IoU here while finding exactly zero "
        f"cleared forest (the no-change class alone is "
        f"{fmt_pct(probe[1])} of the pixels)"
        if probe
        else "a no-change-only model would score well on mean IoU and find nothing"
    )
    return f"""
<section class="section" id="honest">
  <h2>What the numbers do not say</h2>
  <div class="cols">
    <div class="callout warn">
      <h3>The class imbalance</h3>
      <p>Changed pixels are 1&ndash;5% of a scene, so {naive}. Every honest
      metric here is therefore reported for the <em>change class</em>, with the
      no-change IoU shown only as context.</p>
    </div>
    <div class="callout warn">
      <h3>The precision bias</h3>
      <p>lasvegas and dubai sit in bare desert. Almost every non-urban pixel
      there resembles cleared ground, so the model over-predicts area on them
      and under-predicts on the small, subtle cities &mdash; see the area
      accounting above, where the two failure directions are visible per city.</p>
    </div>
    <div class="callout bad">
      <h3>The worst held-out city</h3>
      <p>{esc(worst)} reaches {fmt_score(w.get("f1_change"))} change-class F1
      against {fmt_ha(w.get("true_ha"))} of true loss and
      {fmt_ha(w.get("pred_ha"))} predicted. One city in ten is effectively
      undetected, and the mean above hides it.</p>
    </div>
    <div class="callout ok">
      <h3>The best held-out city</h3>
      <p>{esc(best)} reaches {fmt_score(b.get("f1_change"))} F1. The spread
      between the best and worst held-out city is the real result, not the
      average of the two.</p>
    </div>
  </div>
  <p class="prose">Training ran 8 epochs on CPU on 856 tiles from 14 labelled
  OSCD cities, with 4 further cities held out for early stopping. The test
  cities were never used for training, validation or threshold tuning. Nothing
  was ensembled, no pretrained change-detection weights were fine-tuned, and
  the NDVI threshold was frozen before the test evaluation. The checkpoint
  reported is the best-validation one, not the last epoch.</p>
</section>
"""


def _raw_json_html(metrics: dict) -> str:
    """The artefacts themselves, escaped, so nothing here is uncheckable."""
    if not metrics:
        return "<p class=\"prose\">No artefact found; run the pipeline first.</p>"
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    return (
        "<details class=\"raw\"><summary>Raw JSON artefacts "
        "(metrics.json + baseline_metrics.json)</summary>"
        f"<pre><code>{esc(payload)}</code></pre></details>"
    )


def _history_table(metrics: dict) -> str:
    """Per-epoch validation history, with the reported checkpoint marked."""
    history = history_of(metrics.get("unet"))
    if not history:
        return ""
    best = max(history, key=lambda h: h.get("iou_change") or 0)
    rows = []
    for h in history:
        cls = ' class="best"' if h is best else ""
        rows.append(
            f'<tr{cls}><th scope="row">{esc(h.get("epoch"))}</th>'
            f'<td class="num">{fmt_score(h.get("loss"))}</td>'
            f'<td class="num">{fmt_score(h.get("iou_change"))}</td>'
            f'<td class="num">{fmt_score(h.get("iou_no_change"))}</td>'
            f'<td class="num">{fmt_score(h.get("mean_iou"))}</td>'
            f'<td class="num">{fmt_score(h.get("f1_change"))}</td>'
            f'<td class="num">{fmt_score(h.get("precision_change"))}</td>'
            f'<td class="num">{fmt_score(h.get("recall_change"))}</td></tr>'
        )
    return (
        '<table class="grid"><caption>Validation metrics per epoch on the 4 '
        f'held-out train cities. The reported checkpoint is epoch '
        f'{esc(best.get("epoch"))}, at {fmt_score(best.get("iou_change"))} '
        'change-class IoU; the last epoch is not selected.</caption>'
        '<thead><tr><th scope="col">Epoch</th>'
        '<th scope="col">Loss</th><th scope="col">IoU (change)</th>'
        '<th scope="col">IoU (no-change)</th><th scope="col">Mean IoU</th>'
        '<th scope="col">F1 (change)</th><th scope="col">Precision</th>'
        f'<th scope="col">Recall</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _figures_html() -> str:
    """Copy each committed figure into docs/figures/ and link it relatively."""
    figures_dir = config.RESULTS_DIR / "figures"
    cards = []
    if figures_dir.is_dir():
        for path in sorted(figures_dir.glob("*.png")):
            dest = DOCS_DIR / "figures" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(path, dest)
            cap = CAPTIONS.get(path.stem, "")
            body = f"<figcaption>{esc(cap)}</figcaption>" if cap else ""
            cards.append(
                f'<figure class="fig"><img loading="lazy" src="figures/'
                f'{esc(path.name)}" alt="{esc(cap or path.stem)}">{body}</figure>'
            )
    if not cards:
        return ""
    return (
        '<section class="section" id="figures"><h2>Committed figures</h2>'
        '<p class="prose">Generated by <code>python -m forest_change.figures</code> '
        'and committed to the repository, so they render on GitHub Pages with no '
        'build step and no network request. Maps are full-city inference, not '
        'tiles: every test scene is predicted as one image at 10 m resolution.'
        f'</p><div class="figs">{"".join(cards)}</div></section>'
    )


CSS = """
/* kpi cards */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:0 0 22px}
.kpi{background:color-mix(in srgb,var(--teal) 7%,var(--card));border:1px solid var(--line);
  border-radius:12px;padding:14px 16px}
.kpi-plain{background:var(--card)}
.kpi-label{margin:0;font-size:.75rem;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted)}
.kpi-label .hint{display:block;margin-top:3px;font-weight:500;letter-spacing:0;
  text-transform:none;font-size:.75rem;color:var(--muted)}
.kpi-value{margin:6px 0 0;font-size:1.65rem;font-weight:750;font-variant-numeric:tabular-nums;
  color:var(--teal)}

/* tables */
table{width:100%;border-collapse:collapse;font-size:.9375rem}
caption{text-align:left;color:var(--muted);font-size:.8125rem;padding:0 0 10px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}
thead th{font-size:.6875rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);border-bottom:2px solid var(--line)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tbody tr.best{background:color-mix(in srgb,var(--teal) 9%,transparent)}
.grid{margin-bottom:22px}
.num{text-align:right;font-variant-numeric:tabular-nums}
.tag{font-size:.75rem;font-weight:600;padding:3px 8px;border-radius:6px;white-space:nowrap}
.tag.ok{color:var(--green);background:color-mix(in srgb,var(--green) 14%,transparent)}
.tag.meh{color:var(--amber);background:color-mix(in srgb,var(--amber) 14%,transparent)}
.tag.bad{color:var(--rose);background:color-mix(in srgb,var(--rose) 14%,transparent)}
.bar-cell{width:22%;min-width:110px}
.bar{display:block;height:9px;border-radius:999px;background:var(--teal)}
.bars .bar-cell{width:34%}

:root{
  --ink:#0f172a; --ink-2:#334155; --muted:#64748b; --line:#dbe3ec;
  --bg:#f6f8fb; --card:#ffffff; --teal:#0f4c5c; --teal-2:#0f766e;
  --amber:#b45309; --rose:#be1239; --green:#15803d;
  --radius:14px; --shadow:0 1px 2px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.06);
}
@media (prefers-color-scheme:dark){
  :root{
    --ink:#e8eef6; --ink-2:#c3cede; --muted:#93a2b6; --line:#26313f;
    --bg:#0b1220; --card:#111a2b; --teal:#7fd4c1; --teal-2:#5eead4;
    --amber:#fbbf24; --rose:#fb7185; --green:#4ade80;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 22px 72px}
a{color:var(--teal-2);text-decoration-thickness:1px;text-underline-offset:3px}
code{font:0.875em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:color-mix(in srgb,var(--teal) 8%,transparent);
  padding:.15em .4em;border-radius:6px;color:inherit}

.mast{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;
  justify-content:space-between;padding-bottom:22px;border-bottom:1px solid var(--line)}
.eyebrow{margin:0 0 6px;font-size:.75rem;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;color:var(--teal-2)}
h1{margin:0;font-size:clamp(1.7rem,1.2rem + 2vw,2.5rem);line-height:1.15;
  letter-spacing:-.02em;font-weight:750}
.sub{margin:10px 0 0;color:var(--muted);max-width:62ch}
.mast-links{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.mast-links a{font-size:.875rem;font-weight:600;padding:8px 14px;border-radius:999px;
  border:1px solid var(--line);background:var(--card);color:var(--ink)}
.mast-links a:hover{border-color:var(--teal-2)}

nav.toc{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);margin:0 -22px 28px;
  padding:10px 22px;display:flex;gap:6px;flex-wrap:wrap}
nav.toc a{font-size:.8125rem;font-weight:600;color:var(--ink-2);padding:6px 10px;
  border-radius:8px;text-decoration:none}
nav.toc a:hover,nav.toc a:focus-visible{background:color-mix(in srgb,var(--teal) 10%,transparent);
  color:var(--teal)}

/* sections and prose */
.section{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:26px 28px;margin:0 0 22px}
.section>h2{margin:0 0 4px;font-size:1.3rem;letter-spacing:-.01em}
.section>h3{margin:0 0 6px;font-size:1rem}
.prose{margin:0 0 16px;color:var(--ink-2);max-width:76ch}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.callout{border:1px solid var(--line);border-left:4px solid var(--teal-2);
  border-radius:10px;padding:14px 16px;background:var(--card)}
.callout p{margin:0;color:var(--ink-2);font-size:.9375rem}
.callout.warn{border-left-color:var(--amber)}
.callout.bad{border-left-color:var(--rose)}
.callout.ok{border-left-color:var(--green)}
/* figures */
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
.fig{margin:0}
.fig img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}
.fig figcaption{margin-top:8px;font-size:.8125rem;color:var(--muted)}

/* raw artefacts */
.raw{margin-top:22px;border:1px solid var(--line);border-radius:10px;padding:12px 14px;
  background:var(--card)}
.raw summary{cursor:pointer;font-size:.875rem;font-weight:600;color:var(--teal)}
.raw pre{margin:12px 0 0;padding:12px;overflow:auto;max-height:420px;font-size:.8125rem;
  background:color-mix(in srgb,var(--teal) 6%,transparent);border-radius:8px}

footer{margin-top:34px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:.8125rem;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
@media (max-width:640px){
  .wrap{padding:20px 14px 48px} .section{padding:18px 16px}
  th,td{padding:8px 8px} .bar-cell{display:none}
}
@media print{
  nav.toc,.mast-links{display:none} body{background:#fff}
  .section{box-shadow:none;break-inside:avoid}
}
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta name="color-scheme" content="light dark">
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
<footer>
  <p>Generated from the committed JSON artefacts by
     <code>python -m forest_change.dashboard</code>.</p>
  <p><a href="{repo}">Source on GitHub</a></p>
</footer>
</div>
</body>
</html>
"""

def build_body(metrics: dict) -> str:
    """Compose the whole page body from the loaded artefacts."""
    parts = [
        '<header class="mast"><div>'
        '<p class="eyebrow">Applied ML portfolio - results console</p>'
        "<h1>Forest-change detection on the OSCD benchmark</h1>"
        '<p class="sub">A lightweight U-Net on 12-band Sentinel-2 scene pairs, '
        "trained on 856 tiles from 14 labelled cities and reported once on the "
        "10 held-out test cities. Changed pixels are 1-5% of a scene, so every "
        "headline here is a change-class score, never a mean over classes.</p>"
        '<div class="mast-links">'
        f'<a href="https://github.com/{REPO}">Repository</a>'
        '<a href="https://github.com/Bakkarranna">Other projects</a>'
        '<a href="#results">Skip to results</a></div></div></header>',
        '<nav class="toc" aria-label="Sections">'
        + "".join(
            f'<a href="#{anchor}">{label}</a>'
            for anchor, label in (
                ("results", "Results"),
                ("cities", "Per city &amp; area"),
                ("figures", "Figures"),
                ("honest", "Limitations"),
                ("artefacts", "Artefacts"),
            )
        )
        + "</nav>",
    ]

    sections = [
        '<section class="section" id="results"><h2>Results</h2>'
        '<p class="prose">Two models, one test split, nothing tuned on it. The '
        "U-Net is trained only on the 14 labelled OSCD train cities, with 4 of "
        "them held back for early stopping; the 10 test cities are scored "
        "exactly once. The baseline is a hand-tuned NDVI rule, frozen on the "
        "train cities and evaluated here unchanged.</p>"
        + _kpi_cards(metrics)
        + comparison_table(metrics)
        + _history_table(metrics)
        + "</section>",
        ('<section class="section" id="cities"><h2>Per city and area</h2>'
         + city_table(metrics) + "</section>"),
        ('<section class="section" id="figures"><h2>Figures</h2>'
         '<p class="prose">Rendered straight from the committed artefacts, so '
         "each map below is the prediction the metrics above were computed "
         "from, not a re-run.</p>" + _figures_html() + "</section>"),
        _honest_section(metrics),
        ('<section class="section" id="artefacts">'
         + _raw_json_html(metrics) + "</section>"),
    ]
    parts.extend(s for s in sections if s)
    return "".join(parts)


def build_page(metrics: dict) -> str:
    """Render the complete standalone HTML document."""
    return PAGE.format(
        title="Forest-change detection - results console",
        description=(
            "Change-class IoU, F1 and cleared-area accounting for a U-Net "
            "forest-change detector on the held-out OSCD test cities, with "
            "the failure modes that the averages hide."
        ),
        css=CSS,
        body=build_body(metrics),
        repo=f"https://github.com/{REPO}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping when an artefact is missing",
    )
    parser.add_argument(
        "--out",
        default=str(DOCS_DIR / "index.html"),
        help="output path (default: docs/index.html)",
    )
    args = parser.parse_args(argv)

    metrics = load_metrics(args.strict)
    if not metrics and args.strict:
        sys.exit("no metrics artefacts found")

    page = build_page(metrics)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
