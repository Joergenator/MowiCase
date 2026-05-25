# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Step 5 — LLM agent demo
#
# Frozen record of the agent answering the five case questions, plus a few
# follow-ups that exercise the `sql` escape hatch. Each cell prints the
# agent's final response and the tool calls it made along the way.
#
# Requires `ANTHROPIC_API_KEY` in the environment (or .env).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import anthropic

from src.agent import ask, SYSTEM_PROMPT, TOOL_SCHEMAS  # noqa: F401 (kept visible)


client = anthropic.Anthropic()


def run(question: str) -> None:
    print(f"Q: {question}\n")
    _, answer = ask(client, [], question, verbose=True)
    print(f"\nA:\n{answer}\n" + "-" * 80)


# %% [markdown]
# ## Case Q1 — Which areas currently show increasing lice pressure?

# %%
run("Which production areas currently show increasing lice pressure? "
    "Use the last 8 weeks of training data and report the top 3 by trend.")


# %% [markdown]
# ## Case Q2 — Which sites have had repeated breaches?

# %%
run("Which sites have had the most repeated breaches since 2022? "
    "Show the top 5 with their PO and breach count.")


# %% [markdown]
# ## Case Q3 — Which production areas have the highest treatment intensity?

# %%
run("Which production areas have the highest treatment intensity? "
    "Compare with and without cleaner-fish (rensefisk) included, "
    "and explain what the difference shows.")


# %% [markdown]
# ## Case Q4 — What patterns are visible before breaches occur?

# %%
run("What patterns are visible in the 4 weeks leading up to a breach? "
    "Compare to the all-time non-breach baseline.")


# %% [markdown]
# ## Case Q5 — Which sites appear most at risk in the coming weeks?

# %%
run("Which sites appear most at risk over the next 12 weeks? "
    "Use the trained LightGBM model and show the top 5 with their predicted "
    "breach probability and PO.")


# %% [markdown]
# ## Bonus follow-ups — the SQL escape hatch

# %%
run("How many sites have at least one breach in 2025?")


# %%
run("What was the warmest week in PO6 in 2024, and what was the mean "
    "FEMALEADULT for the sites in that PO that week?")
