"""Small text utilities."""


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


def has_prefix(text: str, prefixes: list[str]) -> bool:
    """Return True if text starts with any of the given prefixes."""
    return any(text.startswith(p) for p in prefixes)
