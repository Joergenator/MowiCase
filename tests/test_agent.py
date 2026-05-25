"""Tests for the step-5 LLM agent infrastructure.

These tests cover the tools and the DuckDB layer — NOT the live LLM call,
which would need an API key and an external HTTP round trip. The agent
loop itself is a thin Anthropic SDK wrapper; we trust the SDK.
"""
from __future__ import annotations

import pytest

from src.agent_db import describe_schema, get_conn, run_sql
from src.agent_tools import (
    TOOL_REGISTRY, predict_risk, pre_breach_signature, repeat_offenders,
    schema, sql, treatment_intensity, trending_pos,
)


# ---------------------------------------------------------------------------
# DuckDB layer
# ---------------------------------------------------------------------------

class TestDuckDB:
    def test_views_have_no_2026_data(self):
        """The 2026 leakage firewall holds through DuckDB."""
        n = run_sql("SELECT COUNT(*) AS n FROM vlice WHERE YEAR >= 2026")["n"].iloc[0]
        assert n == 0
        n = run_sql("SELECT COUNT(*) AS n FROM vtreatment WHERE YEAR >= 2026")["n"].iloc[0]
        assert n == 0

    def test_views_have_data(self):
        """Both views populated."""
        assert run_sql("SELECT COUNT(*) AS n FROM vlice")["n"].iloc[0] > 100_000
        assert run_sql("SELECT COUNT(*) AS n FROM vtreatment")["n"].iloc[0] > 10_000

    @pytest.mark.parametrize("query", [
        "DELETE FROM vlice",
        "DROP TABLE vlice",
        "INSERT INTO vlice VALUES (1)",
        "UPDATE vlice SET BREACH=TRUE",
        "CREATE TABLE foo (x INT)",
        "ALTER TABLE vlice ADD COLUMN x INT",
        "ATTACH 'evil.db' AS evil",
        "COPY vlice TO 'leak.csv'",
        "PRAGMA database_size",
    ])
    def test_run_sql_rejects_writes(self, query):
        with pytest.raises(ValueError):
            run_sql(query)

    def test_run_sql_truncates_to_max_rows(self):
        df = run_sql("SELECT * FROM vlice", max_rows=10)
        assert len(df) == 10

    def test_describe_schema_lists_both_tables(self):
        s = describe_schema()
        assert "vlice" in s and "vtreatment" in s
        for col in ("SITENUMBER", "WEEK_START", "BREACH", "FEMALEADULT"):
            assert col in s["vlice"]["columns"]


# ---------------------------------------------------------------------------
# Named tools — sanity checks on real data
# ---------------------------------------------------------------------------

class TestNamedTools:
    def test_trending_pos_returns_all_13_areas(self):
        r = trending_pos(weeks=8)
        assert len(r["rows"]) == 13
        for row in r["rows"]:
            assert "slope_femaleadult_per_week" in row
            assert isinstance(row["slope_femaleadult_per_week"], float)

    def test_repeat_offenders_respects_min_breaches(self):
        r = repeat_offenders(min_breaches=30, since_year=2022, top_n=10)
        for row in r["rows"]:
            assert row["breach_weeks"] >= 30

    def test_treatment_intensity_excludes_cleanerfish_changes_values(self):
        # year=None (all years) — rensefisk only exists 2012-2018 in source data,
        # so a single-year filter can give a no-op exclusion. Chart 10's finding
        # about cleaner-fish reshuffling PO1 is on the all-years view.
        with_cf = treatment_intensity(year=None, exclude_cleanerfish=False)["rows"]
        without_cf = treatment_intensity(year=None, exclude_cleanerfish=True)["rows"]
        by_po_with = {r["PRODUCTIONAREA"]: r["treatments_per_active_site"] for r in with_cf}
        by_po_without = {r["PRODUCTIONAREA"]: r["treatments_per_active_site"] for r in without_cf}
        for po, v_without in by_po_without.items():
            assert v_without <= by_po_with[po] + 1e-9
        assert any(by_po_without[po] < by_po_with[po] - 0.01 for po in by_po_without)

    def test_pre_breach_signature_shows_lift_over_baseline(self):
        r = pre_breach_signature(weeks_before=4)
        pre = r["pre_breach"]["femaleadult"]
        base = r["baseline_non_breach"]["femaleadult"]
        # The whole point of the chart — lice are visibly higher before a breach.
        assert pre > base * 1.5


# ---------------------------------------------------------------------------
# predict_risk — model + feature wiring
# ---------------------------------------------------------------------------

class TestPredictRisk:
    @pytest.mark.parametrize("h", [1, 2, 12])
    def test_returns_valid_probabilities(self, h):
        r = predict_risk(horizon=h, top_n=5)
        assert r["horizon_weeks"] == h
        assert r["model_version"] == "lightgbm_v1"
        assert 0 < len(r["top_sites"]) <= 5
        for site in r["top_sites"]:
            p = site["predicted_breach_probability"]
            assert 0.0 <= p <= 1.0

    def test_predict_from_week_is_within_training_range(self):
        r = predict_risk(horizon=1, top_n=1)
        # The latest week scored must be in our cleaned training range (< 2026).
        assert r["predict_from_week"] < "2026-01-01"

    def test_rejects_invalid_horizon(self):
        with pytest.raises(ValueError):
            predict_risk(horizon=99)


# ---------------------------------------------------------------------------
# Tool registry coverage
# ---------------------------------------------------------------------------

def test_tool_registry_matches_module_exports():
    expected = {"trending_pos", "repeat_offenders", "treatment_intensity",
                "pre_breach_signature", "predict_risk", "sql", "schema",
                "read_findings", "list_research_sites"}
    assert set(TOOL_REGISTRY) == expected


def test_read_findings_returns_nonempty():
    from src.agent_tools import read_findings
    r = read_findings()
    assert "content" in r and len(r["content"]) > 500
    assert "breach" in r["content"].lower()


# ---------------------------------------------------------------------------
# Research-site filtering — Sauaneset I (13035) must be excluded by default
# ---------------------------------------------------------------------------

class TestResearchSiteFiltering:
    SAUANESET_I = 13035

    def test_list_research_sites_returns_all_nine(self):
        from src.agent_tools import list_research_sites
        r = list_research_sites()
        assert r["n_sites"] == 9
        ids = {s["site_number"] for s in r["sites"]}
        assert self.SAUANESET_I in ids

    def test_predict_risk_excludes_research_by_default(self):
        r = predict_risk(horizon=12, top_n=20)
        assert r["commercial_only"] is True
        assert r["research_sites_excluded"] == 9
        site_ids = {s["SITENUMBER"] for s in r["top_sites"]}
        assert self.SAUANESET_I not in site_ids

    def test_repeat_offenders_excludes_research_by_default(self):
        r = repeat_offenders(min_breaches=50, since_year=2012, top_n=20)
        assert r["commercial_only"] is True
        site_ids = {row["SITENUMBER"] for row in r["rows"]}
        assert self.SAUANESET_I not in site_ids

    def test_load_data_opt_out_brings_back_hi_sites(self):
        """The commercial_only=False opt-out is still available for audit /
        verification queries — research-site rows reappear in the raw load."""
        from src.load_data import load_lice
        full = load_lice(commercial_only=False)
        commercial = load_lice(commercial_only=True)
        assert self.SAUANESET_I in full["SITENUMBER"].unique()
        assert self.SAUANESET_I not in commercial["SITENUMBER"].unique()
        # Removal accounts for all rows belonging to the 4 HI sites that
        # actually appear in the BarentsWatch dataset.
        assert len(full) > len(commercial)


def test_sql_tool_returns_records_shape():
    r = sql("SELECT COUNT(*) AS n FROM vlice")
    assert r["row_count"] == 1
    assert r["columns"] == ["n"]
    assert r["rows"][0]["n"] > 100_000


def test_schema_tool_passthrough():
    assert schema() == describe_schema()
