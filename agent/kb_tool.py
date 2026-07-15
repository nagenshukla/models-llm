"""
Grounded lookup tool over the hand-curated BIM knowledge base
(data/kb_bim.py, data/structured_facts.py).

This is the anti-hallucination mechanism for exact/structured facts: rather
than trusting the fine-tuned model to recite ISO 19650's part numbers or
the LOD scale correctly from memory, the agent (see bim_agent.py) is
instructed to call lookup_bim_term() and quote its result for any defined
term, standard, or numbered fact. The model still handles conversation,
phrasing, and reasoning - it just isn't the source of truth for specifics
it might paraphrase wrong.
"""

import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from kb_bim import KB, KB_BY_ID  # noqa: E402
from structured_facts import STRUCTURED_FACTS  # noqa: E402

_TERM_NAMES = {entry["term"].lower(): entry["id"] for entry in KB}
_ACRONYMS = {entry["acronym"].lower(): entry["id"] for entry in KB if entry["acronym"]}


def list_bim_terms() -> list[str]:
    """Returns every term name in the knowledge base, for discovery/browsing."""
    return [entry["term"] for entry in KB]


def lookup_bim_term(term: str) -> str:
    """
    Looks up a BIM/digital-construction term, acronym, or standard in the
    ground-truth knowledge base and returns its authoritative definition
    and example. Always call this before stating specifics (definitions,
    standard part numbers, LOD levels, acronym expansions) about a named
    BIM term - do not answer those from memory alone.

    Args:
        term: The term, acronym, or standard name to look up (e.g. "LOD",
            "ISO 19650", "Common Data Environment", "clash detection").

    Returns:
        The authoritative definition and example, or a not-found message
        with the closest known terms if there's no confident match.
    """
    query = term.strip().lower()

    entry_id = _TERM_NAMES.get(query) or _ACRONYMS.get(query)
    if entry_id is None:
        matches = difflib.get_close_matches(query, list(_TERM_NAMES.keys()), n=1, cutoff=0.6)
        if matches:
            entry_id = _TERM_NAMES[matches[0]]

    if entry_id is not None:
        entry = KB_BY_ID[entry_id]
        result = f"{entry['term']}"
        if entry["acronym"]:
            result += f" ({entry['acronym']})"
        result += f": {entry['definition']} Example: {entry['example']}"
        return result

    suggestions = difflib.get_close_matches(query, list(_TERM_NAMES.keys()), n=3, cutoff=0.3)
    if suggestions:
        return f"No exact match for '{term}'. Closest known terms: {', '.join(suggestions)}."
    return f"'{term}' is not in the knowledge base. Do not invent specifics for it - say you're not certain."


def lookup_structured_fact(question: str) -> str:
    """
    Looks up an exact, numbered/structured fact (e.g. "what are the parts
    of ISO 19650", "list the LOD scale", "what are the CDE states") from
    the ground-truth structured facts list. Always call this before
    stating any numbered enumeration - these are exactly the specifics the
    model has been observed inventing incorrectly from memory.

    Args:
        question: The structured-fact question being asked.

    Returns:
        The exact authoritative answer, or a not-found message.
    """
    query = question.strip().lower()
    questions = [q for q, _ in STRUCTURED_FACTS]
    matches = difflib.get_close_matches(query, [q.lower() for q in questions], n=1, cutoff=0.4)
    if matches:
        idx = [q.lower() for q in questions].index(matches[0])
        return STRUCTURED_FACTS[idx][1]
    return (
        f"No exact structured fact matches '{question}'. Do not invent numbered specifics - "
        "say you're not certain and suggest checking the source standard."
    )
