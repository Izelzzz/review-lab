import asyncio
import logging
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from review_agents.db import connect
from review_agents.redact import REDACT

log = logging.getLogger(__name__)

ReviewHandler = Callable[[sqlite3.Connection, int], None]


def not_implemented(conn: sqlite3.Connection, review_id: int) -> None:
    raise NotImplementedError("the review pipeline is not implemented yet")


# Claims of one review, counting the first: a review that keeps dying with the service is
# not queued again and again.
MAX_ATTEMPTS = 2


def recover_interrupted(conn: sqlite3.Connection) -> int:
    """Settle reviews a previous process left unfinished; return how many failed.

    A running review that needs no token is queued once more (decision D-021); one with
    a token cannot run again, since the token lived only in that process's memory, so it
    fails, as does a queued one with a token. Running the CI job again starts afresh.
    """
    conn.execute(
        "UPDATE reviews SET status = 'queued', started_at = NULL"
        " WHERE status = 'running' AND has_token = 0 AND attempts < ? AND superseded_by IS NULL",
        (MAX_ATTEMPTS,),
    )
    cursor = conn.execute(
        "UPDATE reviews SET status = 'failed', error = CASE status"
        " WHEN 'running' THEN 'interrupted by a service restart'"
        " ELSE 'its token was lost in a service restart; run the CI job again' END,"
        " finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        " WHERE status = 'running' OR (status = 'queued' AND has_token = 1)"
    )
    return cursor.rowcount


def claim_next(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "UPDATE reviews SET status = 'running', attempts = attempts + 1,"
        " started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
        " WHERE id = (SELECT id FROM reviews WHERE status = 'queued' ORDER BY id LIMIT 1)"
        " RETURNING id"
    ).fetchone()
    return None if row is None else int(row["id"])


def finish(conn: sqlite3.Connection, review_id: int, error: str | None) -> None:
    conn.execute(
        "UPDATE reviews SET status = ?, error = ?,"
        " finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        # Masked: the error is returned to CI, which may post it (issue #15).
        ("failed" if error else "done", REDACT.optional(error), review_id),
    )


def run_once(db_path: Path, handler: ReviewHandler) -> int | None:
    """Claim and process one queued review; return its id, or None if the queue is empty."""
    with closing(connect(db_path, create=False)) as conn:
        review_id = claim_next(conn)
        if review_id is None:
            return None
        error: str | None = None
        try:
            handler(conn, review_id)
        except Exception as exc:
            log.exception("review %s failed", review_id)
            error = f"{type(exc).__name__}: {exc}"
        # Writes the handler left uncommitted are incomplete, and would swallow the status update.
        if conn.in_transaction:
            conn.rollback()
            error = error or "the handler left a transaction open"
        finish(conn, review_id, error)
        return review_id


class Worker:
    """Processes queued reviews one at a time inside the service process."""

    def __init__(self, db_path: Path, handler: ReviewHandler, poll_seconds: float) -> None:
        self._db_path = db_path
        self._handler = handler
        self._poll_seconds = poll_seconds
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        with closing(connect(self._db_path, create=False)) as conn:
            recovered = recover_interrupted(conn)
        if recovered:
            log.warning("marked %d interrupted review(s) as failed", recovered)
        self._task = asyncio.create_task(self._loop())

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except BaseException:
            # The loop already died; shutdown should report it, not fail because of it.
            log.exception("the worker had stopped with an error")

    async def _loop(self) -> None:
        last_failure: str | None = None
        while True:
            try:
                # The handler blocks on I/O and model calls, so it runs off the event loop.
                processed = await asyncio.to_thread(run_once, self._db_path, self._handler)
                last_failure = None
            except Exception as exc:
                # A locked or briefly unavailable database must not stop the queue for good.
                failure = f"{type(exc).__name__}: {exc}"
                if failure != last_failure:
                    log.exception("worker iteration failed; retrying")
                else:
                    log.debug("worker iteration failed again: %s", failure)
                last_failure = failure
                processed = None
            if processed is None:
                await asyncio.sleep(self._poll_seconds)
