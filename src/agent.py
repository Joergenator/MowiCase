"""CLI agent for the Mowi BarentsWatch case (step 5).

A small REPL that wraps Anthropic's tool-use loop around the tools in
`src.agent_tools`. The LLM has read-only access to vlice / vtreatment via
DuckDB plus calibrated LightGBM forward predictions.

Run:
    python -m src.agent                # defaults to haiku (cheap)
    python -m src.agent --model sonnet # stronger reasoning, ~4x cost
    python -m src.agent --model opus   # best reasoning, ~25x cost

Requires ANTHROPIC_API_KEY in the environment (or a .env file in the repo root).

Type `/quit` to exit, `/tools` to list available tools, `/reset` to clear
the conversation history, `/model <name>` to switch model mid-session.

Cost control
------------
- Default model is Haiku 4.5 — roughly 4x cheaper than Sonnet and adequate
  for tool-calling + SQL. Sonnet is a flag-flip away for harder reasoning.
- Anthropic prompt caching is enabled on the system prompt + tool schemas.
  After the first turn within a 5-minute window, those tokens cost 10% of
  normal input price — a big saving on multi-turn conversations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import anthropic

from src.agent_tools import TOOL_REGISTRY


# Model registry — short name -> full model id. The default is intentionally
# the cheapest tool-use-capable model; users can opt up.
MODEL_OPTIONS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}
DEFAULT_MODEL = "haiku"

MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 12

SYSTEM_PROMPT = """\
You are an analyst working with BarentsWatch salmon-lice data for Mowi.

You have read-only access to two tables (via the `sql` tool):
- vlice: weekly per-site lice counts and sea temperature, 2012-2025
- vtreatment: weekly per-site treatment events, 2012-2025

Important data rules:
- BREACH is NULL when a site was not counted that week. Filter `BREACH = TRUE`
  for breaches; never treat NULL as FALSE.
- The regulatory limit `LICELIMITWEEK` switches between 0.5 (normal) and 0.2
  (spring window, ISO weeks ~16-22). A BREACH is FEMALEADULT >= LICELIMITWEEK.
- All data is pre-2026 by design — there is a strict 2026 leakage firewall.
  Do not pretend to have 2026 observations.

What we already know from EDA (training data, pre-2026):
- Lice pressure rises sharply above ~10 C; warm years drive high-lice years
  (cross-year correlation r=0.84 in the W25-40 peak window).
- Breach rates concentrate in PO3 (Karmoy til Sotra), PO4 (Nordhordaland til
  Stadt), PO5 (Stadt til Hustadvika); overall base rate is 4.55% across all
  years 2012-2025 (commercial sites, weeks where lice were counted).
- Excluding cleaner-fish (rensefisk) stocking from "treatment intensity"
  reshuffles the PO ranking — PO1 drops from #1 to #11 because ~46% of its
  treatments are passive biological control.
- Call `read_findings` for the full EDA narrative when a question needs more
  context than the summary above.
- 9 sites are operated by Havforskningsinstituttet (HI) for research, not
  commercial production. They are excluded from ALL data layers (EDA,
  training, predictions, SQL queries) because their elevated lice loads
  reflect study protocols, not commercial operation. If a user asks about
  research sites or wonders why a known site like Sauaneset I is missing
  from a ranking, call `list_research_sites` to show the registry.

Tool strategy:
- For the five common questions, prefer the named tools — they are tested
  and use defensible logic:
    * trending_pos        → "areas with rising lice pressure"
    * repeat_offenders    → "sites with the most breaches"
    * treatment_intensity → "production areas with highest treatment intensity"
    * pre_breach_signature → "what is visible before a breach"
    * predict_risk        → "which sites are most at risk in the coming weeks"
- For anything else, call `schema` first to see column names, then write a
  SELECT against vlice / vtreatment via the `sql` tool. SQL is DuckDB dialect.
- Use `read_findings` when the user asks about patterns, trends, or
  interpretation that EDA has already characterised.
- Be concise. Cite the numbers you used. When you call `predict_risk`, mention
  the model version, horizon, and the week you predicted from.
- Plain text only. Do not use emojis (including medal/award symbols, arrows, etc.).
  Use markdown tables and bold sparingly.
"""


# ---------------------------------------------------------------------------
# Tool schemas — Anthropic tool_use format
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "trending_pos",
        "description": "Slope of weekly mean adult-female lice (and breach rate) per "
                       "production area over the last N weeks. Use for 'which areas "
                       "currently show increasing lice pressure'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weeks": {"type": "integer", "description": "Lookback window in weeks (default 8)."},
            },
        },
    },
    {
        "name": "repeat_offenders",
        "description": "Sites with the most breach weeks since a given year. "
                       "HI research sites are excluded at the data layer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "min_breaches": {"type": "integer", "description": "Minimum breach weeks to include (default 5)."},
                "since_year": {"type": "integer", "description": "Earliest year to count (default 2020)."},
                "top_n": {"type": "integer", "description": "How many sites to return (default 20)."},
            },
        },
    },
    {
        "name": "treatment_intensity",
        "description": "Treatments per active site per PO. Set exclude_cleanerfish=true "
                       "to focus on active interventions (excluding passive rensefisk stocking).",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Restrict to a specific year (optional)."},
                "exclude_cleanerfish": {"type": "boolean", "description": "Exclude rensefisk events (default true)."},
            },
        },
    },
    {
        "name": "pre_breach_signature",
        "description": "Mean FEMALEADULT / MOBILELICE / SEATEMP in the N weeks before a "
                       "breach vs. the baseline (non-breach weeks).",
        "input_schema": {
            "type": "object",
            "properties": {
                "weeks_before": {"type": "integer", "description": "How many weeks to look back from each breach (default 4)."},
            },
        },
    },
    {
        "name": "predict_risk",
        "description": "Score the latest available week per site with the trained "
                       "LightGBM model. Returns the top_n sites by predicted breach "
                       "probability for the chosen horizon. HI research sites are "
                       "excluded at the data layer; the booster is trained on "
                       "commercial-only data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "horizon": {"type": "integer", "enum": [1, 2, 12], "description": "Forecast horizon in weeks: 1, 2, or 12."},
                "top_n": {"type": "integer", "description": "How many sites to return (default 20)."},
            },
        },
    },
    {
        "name": "sql",
        "description": "Run a read-only SELECT query against vlice and/or vtreatment "
                       "(DuckDB dialect). Use this for ad-hoc questions not covered by "
                       "the named tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A DuckDB SELECT query."},
                "max_rows": {"type": "integer", "description": "Row limit (default 200)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "schema",
        "description": "Return the column dictionary for vlice and vtreatment. "
                       "Call this BEFORE writing SQL so column names are correct.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_findings",
        "description": "Return the full EDA findings document — numbered narrative "
                       "of what we learned from the data (treatment intensity by PO, "
                       "temperature-lice relationship, seasonality, geographic "
                       "clustering, etc.). Call when the user asks for interpretation, "
                       "patterns, or background context the SQL data alone cannot provide.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_research_sites",
        "description": "Return the 9 Havforskningsinstituttet (HI) research aquaculture "
                       "sites — these are excluded by default from predict_risk and "
                       "repeat_offenders. Call when the user asks which sites are "
                       "research, why a particular site is missing from a ranking, or "
                       "what HI's facilities are.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _run_tool(name: str, args: dict) -> str:
    """Execute a registered tool, returning a JSON string."""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_REGISTRY[name](**args)
        return json.dumps(result, default=str)
    except Exception as exc:  # noqa: BLE001 — surface failures to the model
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _cached_system_and_tools() -> tuple[list, list]:
    """Wrap system prompt + tool schemas with cache_control for prompt caching.

    Marking the final tool schema as `ephemeral` caches everything before it
    (system prompt + all preceding tools) in a single cache block. After the
    first turn within the cache TTL (5 min default), those tokens cost ~10%
    of the normal input price.
    """
    system_blocks = [{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]
    cached_tools = [*TOOL_SCHEMAS[:-1],
                    {**TOOL_SCHEMAS[-1], "cache_control": {"type": "ephemeral"}}]
    return system_blocks, cached_tools


def ask(client: anthropic.Anthropic, history: list, user_text: str,
        *, model: str = MODEL_OPTIONS[DEFAULT_MODEL],
        verbose: bool = True) -> tuple[list, str]:
    """Send `user_text` plus tool-use loop, return (new_history, final_text).

    `model` accepts a full Anthropic model id (e.g. 'claude-haiku-4-5').
    """
    history = history + [{"role": "user", "content": user_text}]
    system_blocks, cached_tools = _cached_system_and_tools()

    for round_idx in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_blocks,
            tools=cached_tools,
            messages=history,
        )
        history = history + [{"role": "assistant", "content": resp.content}]

        if resp.stop_reason != "tool_use":
            # Plain text response — extract and return
            final = "".join(b.text for b in resp.content if b.type == "text")
            return history, final.strip()

        # Execute each tool_use block, send back tool_result
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"  · calling {block.name}({json.dumps(block.input)})")
            result = _run_tool(block.name, dict(block.input))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        history = history + [{"role": "user", "content": tool_results}]

    return history, "[stopped — exceeded tool-use round limit]"


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

BANNER = """\
Mowi lice agent ({model}) — /quit to exit, /reset to clear history,
                            /tools to list tools, /model <name> to switch.
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Mowi lice agent CLI")
    parser.add_argument("--model", choices=list(MODEL_OPTIONS), default=DEFAULT_MODEL,
                        help=f"Which Claude model to use (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()
    model_name = args.model
    model_id = MODEL_OPTIONS[model_name]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. "
              "Add it to your environment or .env file.")
        sys.exit(1)

    client = anthropic.Anthropic()
    history: list = []
    print(BANNER.format(model=f"{model_name} = {model_id}"))

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            break
        if user_text == "/reset":
            history = []
            print("(history cleared)")
            continue
        if user_text == "/tools":
            for t in TOOL_SCHEMAS:
                print(f"  {t['name']}: {t['description'].splitlines()[0]}")
            continue
        if user_text.startswith("/model"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 2 and parts[1] in MODEL_OPTIONS:
                model_name = parts[1]
                model_id = MODEL_OPTIONS[model_name]
                print(f"(model switched to {model_name} = {model_id})")
            else:
                print(f"usage: /model {{{','.join(MODEL_OPTIONS)}}}")
            continue

        history, final = ask(client, history, user_text, model=model_id, verbose=True)
        print(final + "\n")


if __name__ == "__main__":
    main()
