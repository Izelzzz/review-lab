"""Small text utilities."""

import re
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in text."""
    return len(text.split())


def truncate(text: str, limit: int) -> str:
    """Shorten text to at most `limit` characters, ending with '...' when cut."""
    if limit < 3:
        raise ValueError("limit must be at least 3")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def render_template(name: str, values: dict[str, str]) -> str:
    """Load templates/<name> and replace each {key} with values[key].

    Placeholders without a matching key are left as they are.
    """
    template = (TEMPLATE_DIR / name).read_text(encoding="utf-8")
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)
