"""Tests for the static results console in ``forest_change.dashboard``.

They exercise the renderers against the committed artefacts, so a formatting
regression or a broken ``docs/`` layout is caught without opening a browser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forest_change import dashboard  # noqa: E402


@pytest.fixture(scope="module")
def metrics() -> dict:
    return dashboard.load_metrics()


@pytest.fixture(scope="module")
def cities(metrics) -> dict:
    return dashboard.per_city(metrics.get("unet"))


def test_means_are_unweighted_over_cities(cities):
    means = dashboard.mean_over_cities(cities)
    manual = sum(c["f1_change"] for c in cities.values()) / len(cities)
    assert means["f1_change"] == pytest.approx(manual)
    assert len(means) >= len(dashboard.MEAN_KEYS) - 1


def test_naive_probe_derives_a_no_change_only_score(cities):
    probe = dashboard.naive_probe(cities)
    assert probe is not None
    mean_iou, share = probe
    # Predicting "no change" everywhere scores mean IoU = no-change share / 2.
    assert mean_iou == pytest.approx(share / 2)
    # The point of the probe: a good-looking number that finds nothing.
    assert mean_iou > 0.45


def test_kpi_cards_show_the_baseline_margin(metrics):
    cards = dashboard._kpi_cards(metrics)
    assert cards.count('class="kpi"') + cards.count('class="kpi kpi-plain"') == 4
    assert "Margin over baseline" in cards
    assert "f1_change" in json.dumps(dashboard.per_city(metrics["baseline"]))


def test_comparison_table_has_one_row_per_metric(metrics):
    table = dashboard.comparison_table(metrics)
    for label in (
        "IoU (change class)",
        "F1 (change class)",
        "Precision (change class)",
        "Recall (change class)",
        "Mean IoU (both classes)",
        "IoU (no-change class)",
    ):
        assert label in table
    assert "NDVI-loss baseline" in table


def test_city_table_rows_carry_area_accounting(metrics):
    table = dashboard.city_table(metrics)
    cities = dashboard.per_city(metrics.get("unet"))
    assert table.count("<tr><th") == len(cities) + 1
    # First data row is the best city, so the table reads best-to-worst.
    best = max(cities, key=lambda c: cities[c]["f1_change"])
    assert f">{best}<" in table.split("<tbody>")[1]


def test_history_table_marks_the_reported_checkpoint(metrics):
    table = dashboard._history_table(metrics)
    assert 'class="best"' in table
    best = max(dashboard.history_of(metrics["unet"]), key=lambda h: h["iou_change"])
    assert f'>{best["epoch"]}</th>' in table


def test_honest_section_names_the_spread_not_the_mean(metrics):
    body = dashboard._honest_section(metrics)
    cities = dashboard.per_city(metrics["unet"])
    worst = min(cities, key=lambda c: cities[c]["f1_change"])
    best = max(cities, key=lambda c: cities[c]["f1_change"])
    assert worst in body and best in body
    assert "no-change" in body


def test_raw_json_round_trips_the_artefacts(metrics):
    block = dashboard._raw_json_html(metrics)
    assert "metrics.json + baseline_metrics.json" in block
    # The escaped payload appears exactly once: no truncation, no duplication.
    assert block.count(dashboard.esc(json.dumps(metrics, indent=2, sort_keys=True))) == 1


def test_figures_are_copied_beside_the_page(tmp_path, metrics, monkeypatch):
    """Every committed figure lands in docs/figures with a captioned <img>."""
    monkeypatch.setattr(dashboard, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(dashboard.config, "RESULTS_DIR", dashboard.config.RESULTS_DIR)
    before = len(list((tmp_path / "figures").glob("*.png"))) if (tmp_path / "figures").is_dir() else 0
    body = dashboard._figures_html()
    copied = list((tmp_path / "figures").glob("*.png"))
    committed = list((dashboard.config.RESULTS_DIR / "figures").glob("*.png"))
    assert len(copied) == len(committed) > 0
    assert len(copied) > before
    assert 'src="figures/' in body
    for path in committed:
        assert dashboard.CAPTIONS.get(path.stem, "x")[:20] in body
        assert (tmp_path / "figures" / path.name).read_bytes() == path.read_bytes()


def test_figures_html_is_idempotent(tmp_path, monkeypatch):
    """A re-run must not duplicate or lose figures in docs/."""
    monkeypatch.setattr(dashboard, "DOCS_DIR", tmp_path)
    dashboard._figures_html()
    first = {p.name: p.read_bytes() for p in (tmp_path / "figures").glob("*.png")}
    assert dashboard._figures_html() == dashboard._figures_html()
    second = {p.name: p.read_bytes() for p in (tmp_path / "figures").glob("*.png")}
    assert first == second


def test_build_page_is_one_self_contained_document(metrics):
    page = dashboard.build_page(metrics)
    assert page.startswith("<!doctype html>")
    assert page.endswith("</html>\n")
    assert "<style>" in page and "http" not in page.split("<style>")[1].split("</style>")[0]
    # No remote assets: nothing to load, so it renders from file:// and Pages alike.
    assert 'src="http' not in page and "<script" not in page
    for anchor in ("results", "cities", "figures", "honest", "artefacts"):
        assert f'id="{anchor}"' in page


def test_cli_writes_and_strict_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(dashboard, "DOCS_DIR", tmp_path)
    out = tmp_path / "index.html"
    assert dashboard.main(["--out", str(out)]) == 0
    assert "Forest-change detection" in out.read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out

    missing = tmp_path / "nope"
    monkeypatch.setattr(dashboard, "METRICS_PATH", missing)
    monkeypatch.setattr(dashboard, "BASELINE_PATH", missing)
    with pytest.raises(SystemExit):
        dashboard.main(["--out", str(tmp_path / "x.html"), "--strict"])
