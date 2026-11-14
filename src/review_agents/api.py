import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, closing
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from review_agents import __version__
from review_agents.auth import SIGNATURE_HEADER, TIMESTAMP_HEADER, verify
from review_agents.config import Settings
from review_agents.credentials import TokenVault
from review_agents.db import connect, migrate
from review_agents.mirror import check_commit_sha, check_repo_name
from review_agents.redact import REDACT
from review_agents.repo_config import check_branch_name
from review_agents.worker import ReviewHandler, Worker, not_implemented

# A PR description is at most 65,536 characters on GitHub; this leaves room for the rest.
MAX_BODY_BYTES = 1024 * 1024


class ReviewRequest(BaseModel):
    """What the CI job sends to start a review (architecture §12.2)."""

    model_config = ConfigDict(extra="forbid")

    repo: str
    pr: int = Field(ge=1, le=2**31 - 1, strict=True)
    base_sha: str
    head_sha: str
    default_branch: str = Field(min_length=1, max_length=255)
    title: str = Field(max_length=1024)
    # GitHub sends null for a PR without a description.
    body: str | None = None
    event: Literal["opened", "synchronize", "reopened", "manual"]
    # The CI job's read-only GITHUB_TOKEN; held in memory only, never stored.
    token: str | None = Field(default=None, repr=False)

    # The messages name what is wrong, never the submitted value.
    @field_validator("repo")
    @classmethod
    def _repo(cls, value: str) -> str:
        return _quiet(check_repo_name, value, "not an owner/name repository")

    @field_validator("base_sha", "head_sha")
    @classmethod
    def _sha(cls, value: str) -> str:
        return _quiet(check_commit_sha, value, "not a full 40-character commit sha")

    @field_validator("default_branch")
    @classmethod
    def _branch(cls, value: str) -> str:
        # git never creates a branch starting with "-"; refuse it before it nears a command.
        if value.startswith("-"):
            raise ValueError("not a branch name")
        return _quiet(check_branch_name, value, "not a branch name")


def _quiet(check: Callable[[str], str], value: str, reason: str) -> str:
    try:
        return check(value)
    except ValueError:
        raise ValueError(reason) from None


# Whatever the repository's own config later says, a review starts in standard mode;
# the pipeline (#9) applies the mode from the default branch's config.
_START_MODE = "standard"
_LIVE = ("queued", "running", "done")


def _supersede(
    conn: sqlite3.Connection, repo: str, pr: int, newer: int, tokens: TokenVault
) -> None:
    """Mark the PR's unfinished older reviews as superseded by `newer` (decision D-010)."""
    rows = conn.execute(
        "UPDATE reviews SET superseded_by = :newer,"
        " error = CASE status WHEN 'queued' THEN :note ELSE error END,"
        " finished_at = CASE status WHEN 'queued'"
        "   THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now') ELSE finished_at END,"
        " status = CASE status WHEN 'queued' THEN 'failed' ELSE status END"
        " WHERE repo = :repo AND pr = :pr AND id != :newer AND superseded_by IS NULL"
        " AND status IN ('queued', 'running') RETURNING id, status",
        {"newer": newer, "repo": repo, "pr": pr, "note": f"superseded by review {newer}"},
    ).fetchall()
    for row in rows:
        if row["status"] == "failed":  # was queued: it will never run, so drop its token
            tokens.discard(int(row["id"]))


def create_app(
    settings: Settings,
    handler: ReviewHandler = not_implemented,
    vault: TokenVault | None = None,
) -> FastAPI:
    tokens = vault or TokenVault()

    def run_review(conn: sqlite3.Connection, review_id: int) -> None:
        try:
            handler(conn, review_id)
        finally:
            tokens.discard(review_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with closing(connect(settings.db_path)) as conn:
            migrate(conn)
        worker = Worker(settings.db_path, run_review, settings.worker_poll_seconds)
        worker.start()
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(title="review-agents", version=__version__, lifespan=lifespan)
    app.state.tokens = tokens

    async def signed_body(request: Request) -> bytes:
        """The raw body of a request signed with the shared secret, or an HTTP error."""
        if settings.shared_secret is None:
            # Fail closed: without a secret no request can be authenticated.
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "shared secret not set")
        # Read with a running count, so a body without Content-Length cannot fill memory
        # before it is authenticated.
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "body too large")
            chunks.append(chunk)
        body = b"".join(chunks)
        valid = verify(
            settings.shared_secret.get_secret_value(),
            request.headers.get(TIMESTAMP_HEADER),
            request.headers.get(SIGNATURE_HEADER),
            request.method,
            request.url.path,
            body,
        )
        if not valid:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing signature")
        return body

    Signed = Annotated[bytes, Depends(signed_body)]

    @app.post("/v1/reviews", status_code=status.HTTP_202_ACCEPTED)
    def submit(body: Signed, response: Response) -> dict[str, Any]:
        try:
            req = ReviewRequest.model_validate_json(body)
        except ValidationError as exc:
            # Field and message only: the submitted values, the token among them, stay out.
            detail = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail) from None
        held: int | None = None
        previous: str | None = None
        with closing(connect(settings.db_path, create=False)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # The latest submission is the PR's current head, even an earlier commit
                # force-pushed back; a stale CI re-run is filtered out by the CI job (#14).
                # A manual `/review` always starts a fresh full review (architecture §7).
                row = (
                    None
                    if req.event == "manual"
                    else conn.execute(
                        "SELECT id, status FROM reviews WHERE repo = ? AND pr = ? AND head_sha = ?"
                        " AND status IN (?, ?, ?) AND superseded_by IS NULL AND event != 'manual'"
                        " ORDER BY id DESC LIMIT 1",
                        (req.repo, req.pr, req.head_sha, *_LIVE),
                    ).fetchone()
                )
                if row is None:
                    row = conn.execute(
                        "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode, default_branch,"
                        " title, body, event, has_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        " RETURNING id, status",
                        (req.repo, req.pr, req.base_sha, req.head_sha, _START_MODE,
                         req.default_branch, req.title, req.body or "", req.event,
                         req.token is not None),
                    ).fetchone()  # fmt: skip
                    held = int(row["id"])
                    _supersede(conn, req.repo, req.pr, held, tokens)
                else:
                    response.status_code = status.HTTP_200_OK
                    # Only a review not yet claimed takes a re-run's fresh token; a running
                    # one already has its token, and a later one could outlive the review.
                    if row["status"] == "queued" and req.token is not None:
                        held = int(row["id"])
                        conn.execute("UPDATE reviews SET has_token = 1 WHERE id = ?", (held,))
                if held is not None:
                    # Held before the commit, so the worker never claims it without its token.
                    previous = tokens.get(held)
                    tokens.put(held, req.token)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                if held is not None:  # the row was not written, so neither is its token
                    if previous is None:
                        tokens.discard(held)
                    else:
                        tokens.put(held, previous)
                raise
        return {"id": int(row["id"]), "status": row["status"]}

    @app.get("/v1/reviews/{review_id}")
    def poll(review_id: int, body: Signed) -> dict[str, Any]:
        with closing(connect(settings.db_path, create=False)) as conn:
            row = conn.execute(
                "SELECT id, repo, pr, base_sha, head_sha, round, mode, status, error, limit_hit,"
                " superseded_by, created_at, started_at, finished_at, report FROM reviews"
                " WHERE id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "no such review")
            findings = conn.execute(
                "SELECT fingerprint, primary_dimension, dimensions, severity, preexisting, file,"
                " line_start, line_end, title, body, basis, verdict, status"
                " FROM findings WHERE review_id = ? ORDER BY id",
                (review_id,),
            ).fetchall()
        result = dict(row)
        # The summary, inline comments and counts CI posts (report design §7).
        result["report"] = json.loads(row["report"]) if row["report"] is not None else None
        if row["superseded_by"] is not None:
            result["status"] = "superseded"
        # Masked too: CI logs what it polls, and a model may have copied a secret into a
        # finding; the stored text is the reviewed code's, as the mirror has it (issue #15).
        result["findings"] = REDACT.every_string(
            [
                {
                    **dict(f),
                    "dimensions": json.loads(f["dimensions"]),
                    "basis": json.loads(f["basis"]),
                    "preexisting": bool(f["preexisting"]),
                }
                for f in findings
            ]
        )
        return result

    @app.get("/healthz")
    def healthz(request: Request, response: Response) -> dict[str, str]:
        problem = None
        try:
            with closing(connect(settings.db_path, create=False)) as conn:
                conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        except sqlite3.Error:
            problem = "database unavailable"
        if problem is None and not request.app.state.worker.alive:
            problem = "worker stopped"
        if problem is not None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": problem, "version": __version__}
        return {"status": "ok", "version": __version__}

    return app
