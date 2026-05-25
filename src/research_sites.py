"""Registry of Norwegian aquaculture sites operated for research, not
commercial production.

Source: Havforskningsinstituttet (Institute of Marine Research, HI) — these
are the 9 aquaculture sites HI owns and operates. They differ from
commercial sites in ways that distort operational analytics:

  - Lice loads may be deliberately elevated for challenge / resistance studies
  - Treatment frequency is governed by study protocol, not regulation
  - Fallow / restocking cycles follow experiments, not production economics

Including them in `predict_risk` or `repeat_offenders` produces false alerts
that no operator at Mowi should act on. Filter at the agent-tools layer
(not at load_data) so the raw data stays untouched for analytical work.

Verified anomaly profile for Sauaneset I (13035) — typical of this list:
  - Avg FEMALEADULT 0.789 (PO3 avg 0.204) → 3.9x
  - Avg MOBILELICE 3.026 (PO3 avg ~0.7) → 4.3x
  - Breach weeks 265 (PO3 avg 19.3) → 13.7x
  - Treatments 71 (PO3 avg 80.5) → 0.88x
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Site registry — (site_number, site_name, location)
# ---------------------------------------------------------------------------

RESEARCH_SITES: dict[int, dict] = {
    23015: {"name": "Floedevigen",            "location": "Arendal"},
    45012: {"name": "Floedevigen Sjoe Vest",  "location": "Arendal"},
    45011: {"name": "Floedevigen Sjoe Oest",  "location": "Arendal"},
    13567: {"name": "Knappen Solheim",        "location": "Masfjorden"},
    10156: {"name": "Matredal",               "location": "Masfjorden"},
    31597: {"name": "Nordnes",                "location": "Bergen"},
    13035: {"name": "Sauaneset I",            "location": "Austevoll"},
    16195: {"name": "Sauaneset II",           "location": "Austevoll"},
    12154: {"name": "Smoerdalen",             "location": "Masfjorden"},
}

# Convenience set for fast membership checks
RESEARCH_SITE_IDS: frozenset[int] = frozenset(RESEARCH_SITES.keys())

# Cite this string in tool outputs / system prompt
SOURCE_NOTE = "Havforskningsinstituttet (Institute of Marine Research, HI)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_research(site_number: int) -> bool:
    """Return True if the given SITENUMBER is in the HI research registry."""
    return int(site_number) in RESEARCH_SITE_IDS


def filter_commercial(df: pd.DataFrame, site_col: str = "SITENUMBER") -> pd.DataFrame:
    """Drop research sites from a DataFrame.

    No-op if the column is missing — caller's responsibility to pass the
    right frame; this keeps the helper composable.
    """
    if site_col not in df.columns:
        return df
    return df[~df[site_col].isin(RESEARCH_SITE_IDS)].copy()


def list_research_sites() -> list[dict]:
    """Return the registry as a list of JSON-serializable records."""
    return [{"site_number": sid, **info} for sid, info in RESEARCH_SITES.items()]
