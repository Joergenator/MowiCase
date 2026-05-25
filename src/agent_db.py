"""DuckDB layer for the LLM agent.

Wraps the cleaned lice + treatment frames (from `src.load_data`) as
in-memory DuckDB tables so the agent can run SQL against them. The
leakage firewall is preserved because the source frames are loaded
through `load_training_data()`.

Two safety guards on `run_sql`:
  1. The query string is screened for write/DDL verbs.
  2. The DuckDB connection itself has no filesystem write access in the
     in-memory mode — DROP/CREATE on the in-memory tables IS still possible
     from raw SQL, hence the textual guard.
"""
from __future__ import annotations

import re
from functools import lru_cache

import duckdb
import pandas as pd

from src.load_data import load_training_data


# A small regex of write verbs we reject. Conservative — better to refuse a
# legitimate SELECT subquery containing one of these words than to silently
# allow mutation. The LLM can always rephrase.
_WRITE_VERBS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|ATTACH|"
    r"DETACH|COPY|EXPORT|IMPORT|PRAGMA|SET|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def get_conn() -> duckdb.DuckDBPyConnection:
    """Return a process-wide DuckDB connection with vlice/vtreatment registered."""
    lice, treat = load_training_data()
    con = duckdb.connect(":memory:")
    con.register("vlice", lice)
    con.register("vtreatment", treat)
    return con


def run_sql(query: str, max_rows: int = 200) -> pd.DataFrame:
    """Execute a read-only SQL query against vlice / vtreatment.

    Raises ValueError if the query contains a write/DDL verb.
    Truncates results to `max_rows` to keep LLM context bounded.
    """
    if _WRITE_VERBS.search(query):
        raise ValueError(
            "Only read-only SELECT queries are allowed. Forbidden verbs: "
            "INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/TRUNCATE/REPLACE/"
            "ATTACH/DETACH/COPY/PRAGMA/SET/GRANT/REVOKE."
        )
    con = get_conn()
    df = con.execute(query).df()
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df


# ---------------------------------------------------------------------------
# Schema documentation — exposed to the LLM via the `describe_schema` tool
# ---------------------------------------------------------------------------

SCHEMA_DOC = {
    "vlice": {
        "description": (
            "Weekly per-site lice observations. One row per (SITENUMBER, "
            "WEEK_START). Cleaned: implausible sensor values nulled, exact "
            "duplicates deduped. Date range 2012-01-02 to 2025-12-22 (no 2026)."
        ),
        "columns": {
            "SITENUMBER": "int — unique site identifier",
            "SITENAME": "str — human-readable site name",
            "PRODUCTIONAREAID": "int 1-13 — production area number (PO)",
            "PRODUCTIONAREA": "str — production area name (e.g. 'Stadt til Hustadvika')",
            "MUNCIPALITY": "str — Norwegian municipality (NOTE: source column typo, not MUNICIPALITY)",
            "MUNCIPALITYNUMBER": "int — municipality code",
            "COUNTY": "str — Norwegian county name",
            "COUNTYNUMBER": "int — county code",
            "LATITUDE": "float — site latitude",
            "LONGITUDE": "float — site longitude",
            "YEAR": "int — ISO year",
            "WEEK": "int 1-53 — ISO week number",
            "WEEK_START": "date — Monday of the ISO week",
            "FEMALEADULT": "float — adult-female lice per fish (the regulated metric)",
            "MOBILELICE": "float — mobile (juvenile) lice per fish",
            "PERSISTENTLICE": "float — sessile (attached) lice per fish",
            "SEATEMPERATURE": "float — sea temperature in °C",
            "LICELIMITWEEK": "float — regulatory limit that week: 0.5 (normal) or 0.2 (spring window)",
            "OVERTHELICELIMITWEEK": "str — raw source flag ('Ja'/'Nei'/'Ukjent'). Prefer BREACH below.",
            "HAVECOUNTEDLICE": "bool nullable — whether a lice count was taken this week",
            "LIKELYNOFISH": "bool nullable — whether site is fallow (no fish)",
            "BREACH": (
                "bool nullable — TRUE if FEMALEADULT >= LICELIMITWEEK this week. "
                "Use BREACH=TRUE to filter for breach weeks; NULL means the week "
                "was not counted (do NOT treat NULL as FALSE)."
            ),
        },
    },
    "vtreatment": {
        "description": (
            "Weekly per-site treatment events. One row per treatment event — "
            "a single (site, week) can have multiple rows if multiple treatments "
            "happened (e.g. compound chemical baths). Date range matches vlice."
        ),
        "columns": {
            "SITENUMBER": "int — joins to vlice.SITENUMBER",
            "SITENAME": "str",
            "PRODUCTIONAREAID": "int 1-13",
            "PRODUCTIONAREA": "str",
            "MUNCIPALITY": "str",
            "MUNCIPALITYNUMBER": "int",
            "COUNTY": "str",
            "COUNTYNUMBER": "int",
            "LATITUDE": "float",
            "LONGITUDE": "float",
            "YEAR": "int",
            "WEEK": "int",
            "WEEK_START": "date",
            "ACTION": (
                "str — top-level treatment category. Values: 'medikamentell' "
                "(chemical), 'mekanisk fjerning' (mechanical removal), "
                "'ikke-medikamentell' (non-medicinal umbrella, 2024+), "
                "'rensefisk' (cleaner-fish stocking)."
            ),
            "TYPEOFTREATMENT": "str — sub-category within ACTION (e.g. 'badbehandling', 'spyling')",
            "ACTIVEINGREDIENT": "str nullable — chemical compound when ACTION='medikamentell' (e.g. 'Cypermetrin')",
            "CLEANERFISH": "str nullable — species when ACTION='rensefisk' (e.g. 'Berggylt', 'Rognkjeks')",
            "SPECIESID": "int nullable — cleaner-fish species code",
            "SCOPE": "str — extent flag ('hele anlegget', 'delvis')",
        },
    },
}


def describe_schema() -> dict:
    """Return the column dictionary for vlice + vtreatment as plain JSON.

    Used by the agent's `describe_schema` tool so the LLM can write correct
    SQL on novel questions without hallucinating column names.
    """
    return SCHEMA_DOC
