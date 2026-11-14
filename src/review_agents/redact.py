"""Secrets are masked in logs, stored errors and what CI gets (issue #15, §15.3).

Masked: every log line, the errors stored for steps and reviews, the report and the
findings a poll returns. Findings are stored as written: their text is the reviewed
code's, which the repository mirror holds anyway.

Two kinds are masked: the values the service knows are secret (its shared secret, the
model and observability keys, each review's GitHub token) and anything in the shape of
a well-known credential (GitHub tokens, `sk-` keys, bearer and basic credentials, the
password in a URL), which also covers one a model quotes from the code it reviews.
"""

import logging
import re
import threading
from collections import deque

from review_agents.config import Settings

MASK = "***"
# Shorter values are left alone: a key set to "test" must not mask every "test".
MIN_LENGTH = 8
# Reviews' tokens kept for masking, newest first: a token expires within hours, and every
# line masked is checked against each one. The service's own secrets are always kept.
RECENT_TOKENS = 1000
PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    # The scheme word stays, so a reader still sees what kind of credential it was.
    re.compile(r"(?i)(?<=\bbearer )[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)(?<=\bbasic )[A-Za-z0-9+/=]{16,}"),
    # user:password in a URL, such as a proxy's.
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
)


class Redactor:
    """Masks the secrets it knows and those in well-known shapes; safe across threads."""

    def __init__(self, recent: int = RECENT_TOKENS) -> None:
        self._known: set[str] = set()
        self._recent: deque[str] = deque(maxlen=recent)
        self._lock = threading.Lock()

    def add(self, value: str | None) -> None:
        """A secret the service holds for as long as it runs: masked until it stops."""
        if value is not None and len(value) >= MIN_LENGTH:
            with self._lock:
                self._known.add(value)

    def add_recent(self, value: str | None) -> None:
        """A short-lived secret, such as a review's token: only the newest are kept."""
        if value is not None and len(value) >= MIN_LENGTH:
            with self._lock:
                if value not in self._recent:
                    self._recent.append(value)

    def __call__(self, text: str) -> str:
        with self._lock:
            # A longer one first, so one secret inside another leaves nothing behind.
            known = sorted({*self._known, *self._recent}, key=len, reverse=True)
        for value in known:
            text = text.replace(value, MASK)
        for pattern in PATTERNS:
            text = pattern.sub(MASK, text)
        return text

    def optional(self, text: str | None) -> str | None:
        return None if text is None else self(text)

    def every_string(self, value: object) -> object:
        """`value` with each string in it masked, through lists and dicts (JSON data)."""
        if isinstance(value, str):
            return self(value)
        if isinstance(value, list):
            return [self.every_string(v) for v in value]
        if isinstance(value, dict):
            return {k: self.every_string(v) for k, v in value.items()}
        return value


# The service's one redactor: what it learns as it runs is masked everywhere.
REDACT = Redactor()


def register_settings(redact: Redactor, settings: Settings) -> None:
    """The secrets the service is configured with."""
    if settings.shared_secret is not None:
        redact.add(settings.shared_secret.get_secret_value())
    for endpoint in (settings.finder_model, settings.verifier_model):
        if endpoint is not None and endpoint.api_key is not None:
            redact.add(endpoint.api_key.get_secret_value())
    for value in settings.observability.headers.values():
        secret = value.get_secret_value()
        redact.add(secret)
        # "Basic <key>" or "Bearer <key>": the key alone may show up too.
        redact.add(secret.split(" ", 1)[-1])


class LogFilter(logging.Filter):
    """Masks a log record's message, its arguments and its traceback."""

    def __init__(self, redact: Redactor = REDACT) -> None:
        super().__init__()
        self._redact = redact

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(record.getMessage())
        record.args = ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redact(record.stack_info)
        return True


def install_log_filter(redact: Redactor = REDACT) -> None:
    """Every handler of the root logger masks what it writes."""
    for handler in logging.getLogger().handlers:
        handler.addFilter(LogFilter(redact))
