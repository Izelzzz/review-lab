import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from review_agents.api import create_app
from review_agents.auth import SIGNATURE_HEADER, TIMESTAMP_HEADER, sign
from review_agents.config import Settings
from review_agents.credentials import TokenVault
from review_agents.db import connect
from review_agents.worker import recover_interrupted

SECRET = "s" * 40
FAKE_TOKEN = "ghs_fake_read_only_token"  # secret-scan: allow
HEAD = "b" * 40
REQUEST = {
    "repo": "octo/shop",
    "pr": 7,
    "base_sha": "a" * 40,
    "head_sha": HEAD,
    "default_branch": "main",
    "title": "Clamp refunds",
    "body": "Refunds never exceed the paid amount.",
    "event": "synchronize",
    "token": FAKE_TOKEN,
}


def never_runs(conn: sqlite3.Connection, review_id: int) -> None:
    raise AssertionError("the worker should not pick reviews up in these tests")


class Client:
    def __init__(self, data_dir: Path, secret: str | None = SECRET) -> None:
        settings = Settings(
            data_dir=data_dir,
            # Long enough that the worker never claims a review during a test.
            worker_poll_seconds=3600,
            shared_secret=SecretStr(secret) if secret else None,
        )
        self.vault = TokenVault()
        self.http = TestClient(create_app(settings, never_runs, self.vault))
        self.db_path = settings.db_path

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        secret: str = SECRET,
        timestamp: int | None = None,
        signed: bool = True,
    ) -> httpx2.Response:
        body = json.dumps(payload).encode() if payload is not None else b""
        stamp = str(int(time.time()) if timestamp is None else timestamp)
        headers = {"Content-Type": "application/json"}
        if signed:
            headers[TIMESTAMP_HEADER] = stamp
            headers[SIGNATURE_HEADER] = sign(secret, stamp, method, path, body)
        return self.http.request(method, path, content=body, headers=headers)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Client]:
    made = Client(tmp_path / "data")
    with made.http:
        yield made


def test_a_signed_review_is_queued_and_can_be_polled(client: Client) -> None:
    created = client.request("POST", "/v1/reviews", REQUEST)

    assert created.status_code == 202
    review_id = created.json()["id"]
    polled = client.request("GET", f"/v1/reviews/{review_id}")
    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "queued"
    assert body["repo"] == "octo/shop" and body["pr"] == 7 and body["head_sha"] == HEAD
    assert body["findings"] == []
    assert body["report"] is None  # only a finished review has one


@pytest.mark.parametrize(
    ("label", "options"),
    [
        ("missing", {"signed": False}),
        ("wrong secret", {"secret": "x" * 40}),
        ("stale", {"timestamp": int(time.time()) - 3600}),
    ],
)
def test_unsigned_or_badly_signed_requests_are_rejected(
    client: Client, label: str, options: dict[str, object]
) -> None:
    for method, path, payload in (
        ("POST", "/v1/reviews", REQUEST),
        ("GET", "/v1/reviews/1", None),
    ):
        response = client.request(method, path, payload, **options)  # type: ignore[arg-type]

        assert response.status_code == 401, label
    with closing(connect(client.db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM reviews").fetchone()[0] == 0


def test_a_signature_for_one_body_does_not_cover_another(client: Client) -> None:
    stamp = str(int(time.time()))
    body = json.dumps(REQUEST).encode()
    other = json.dumps({**REQUEST, "pr": 8}).encode()
    headers = {
        TIMESTAMP_HEADER: stamp,
        SIGNATURE_HEADER: sign(SECRET, stamp, "POST", "/v1/reviews", body),
    }

    response = client.http.post("/v1/reviews", content=other, headers=headers)

    assert response.status_code == 401


def test_without_a_configured_secret_nothing_is_accepted(tmp_path: Path) -> None:
    unconfigured = Client(tmp_path / "data", secret=None)
    with unconfigured.http:
        response = unconfigured.request("POST", "/v1/reviews", REQUEST)

    assert response.status_code == 503


def test_the_token_is_held_in_memory_only(client: Client) -> None:
    review_id = client.request("POST", "/v1/reviews", REQUEST).json()["id"]

    assert client.vault.get(review_id) == FAKE_TOKEN
    raw = b"".join(p.read_bytes() for p in client.db_path.parent.glob("*.sqlite3*"))
    assert FAKE_TOKEN.encode() not in raw
    assert FAKE_TOKEN not in client.request("GET", f"/v1/reviews/{review_id}").text


def test_the_same_commit_submitted_again_returns_the_same_review(client: Client) -> None:
    first = client.request("POST", "/v1/reviews", REQUEST)
    again = client.request("POST", "/v1/reviews", REQUEST)
    newer = client.request("POST", "/v1/reviews", {**REQUEST, "head_sha": "c" * 40})

    assert again.status_code == 200
    assert again.json()["id"] == first.json()["id"]
    assert newer.status_code == 202 and newer.json()["id"] != first.json()["id"]


@pytest.mark.parametrize(
    "change",
    [
        {"repo": "../etc"},
        {"head_sha": "not-a-sha"},
        {"pr": 0},
        {"event": "deleted"},
        {"title": None},
        {"pr": True},
        {"pr": 10**20},
        {"default_branch": "--upload-pack=x"},
        {"default_branch": "a..b"},
    ],
)
def test_malformed_requests_are_refused(client: Client, change: dict[str, object]) -> None:
    response = client.request("POST", "/v1/reviews", {**REQUEST, **change})

    assert response.status_code == 422
    for value in change.values():
        if isinstance(value, str):
            assert value not in response.text  # what was sent is not echoed back


def test_an_oversized_body_is_refused(client: Client) -> None:
    response = client.request("POST", "/v1/reviews", {**REQUEST, "body": "x" * 2_000_000})

    assert response.status_code == 413


def test_an_unknown_review_is_not_found(client: Client) -> None:
    assert client.request("GET", "/v1/reviews/999").status_code == 404


def test_finished_results_survive_a_restart_and_queued_ones_are_failed(tmp_path: Path) -> None:
    first = Client(tmp_path / "data")
    with first.http:
        done = first.request("POST", "/v1/reviews", REQUEST).json()["id"]
        queued = first.request("POST", "/v1/reviews", {**REQUEST, "pr": 8}).json()["id"]
        with closing(connect(first.db_path)) as conn:
            report = {"summary": "s", "comments": [], "counts": {"complete": True}}
            conn.execute(
                "UPDATE reviews SET status = 'done', finished_at = 'x', report = ? WHERE id = ?",
                (json.dumps(report), done),
            )
            conn.execute(
                "INSERT INTO findings (review_id, fingerprint, primary_dimension, severity,"
                " file, line_start, line_end, title, body, evidence, basis)"
                " VALUES (?, 'f1', 'edge_cases', 'important', 'a.py', 3, 4, 't', 'b', 'e', ?)",
                (done, json.dumps([{"kind": "path", "explanation": "x"}])),
            )

    second = Client(tmp_path / "data")
    with second.http:
        finished = second.request("GET", f"/v1/reviews/{done}").json()
        lost = second.request("GET", f"/v1/reviews/{queued}").json()

    assert finished["status"] == "done"
    assert [f["title"] for f in finished["findings"]] == ["t"]
    assert finished["findings"][0]["basis"] == [{"kind": "path", "explanation": "x"}]
    assert finished["report"] == {"summary": "s", "comments": [], "counts": {"complete": True}}
    assert lost["status"] == "failed"
    assert "restart" in lost["error"]


def test_after_a_restart_only_queued_reviews_whose_token_was_lost_fail(tmp_path: Path) -> None:
    first = Client(tmp_path / "data")
    with first.http:
        public = first.request("POST", "/v1/reviews", {**REQUEST, "token": None}).json()["id"]
        private = first.request("POST", "/v1/reviews", {**REQUEST, "pr": 8}).json()["id"]

    with closing(connect(first.db_path)) as conn:
        recover_interrupted(conn)
        status = dict(conn.execute("SELECT id, status FROM reviews").fetchall())

    assert status == {public: "queued", private: "failed"}


def test_a_re_run_without_a_token_keeps_the_token_of_the_queued_review(client: Client) -> None:
    review_id = client.request("POST", "/v1/reviews", REQUEST).json()["id"]

    client.request("POST", "/v1/reviews", {**REQUEST, "token": None})

    assert client.vault.get(review_id) == FAKE_TOKEN


def test_a_re_run_that_brings_a_token_marks_the_queued_review_as_needing_it(
    client: Client,
) -> None:
    review_id = client.request("POST", "/v1/reviews", {**REQUEST, "token": None}).json()["id"]

    client.request("POST", "/v1/reviews", REQUEST)

    assert client.vault.get(review_id) == FAKE_TOKEN
    with closing(connect(client.db_path)) as conn:
        assert conn.execute("SELECT has_token FROM reviews").fetchone()[0] == 1


def test_a_running_review_takes_no_new_token(client: Client) -> None:
    review_id = client.request("POST", "/v1/reviews", REQUEST).json()["id"]
    with closing(connect(client.db_path)) as conn:
        conn.execute("UPDATE reviews SET status = 'running' WHERE id = ?", (review_id,))
    client.vault.discard(review_id)  # as when the review finishes

    client.request("POST", "/v1/reviews", REQUEST)

    assert client.vault.get(review_id) is None


def test_a_pr_without_a_description_is_accepted(client: Client) -> None:
    assert client.request("POST", "/v1/reviews", {**REQUEST, "body": None}).status_code == 202


def test_a_chunked_body_is_cut_off_at_the_limit(client: Client) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(3):
            yield b"x" * 600_000

    response = client.http.post("/v1/reviews", content=chunks())

    assert response.status_code == 413


def test_a_new_push_supersedes_the_older_reviews_of_the_pull_request(client: Client) -> None:
    queued = client.request("POST", "/v1/reviews", REQUEST).json()["id"]
    running = client.request("POST", "/v1/reviews", {**REQUEST, "head_sha": "c" * 40}).json()["id"]
    with closing(connect(client.db_path)) as conn:
        conn.execute("UPDATE reviews SET status = 'running' WHERE id = ?", (running,))
    other_pr = client.request("POST", "/v1/reviews", {**REQUEST, "pr": 8}).json()["id"]

    newest = client.request("POST", "/v1/reviews", {**REQUEST, "head_sha": "d" * 40}).json()["id"]

    # Each review points at the one that replaced it: queued -> running -> newest.
    for old, by in ((queued, running), (running, newest)):
        polled = client.request("GET", f"/v1/reviews/{old}").json()
        assert polled["status"] == "superseded"
        assert polled["superseded_by"] == by
    assert client.request("GET", f"/v1/reviews/{other_pr}").json()["status"] == "queued"
    assert client.request("GET", f"/v1/reviews/{newest}").json()["superseded_by"] is None
    assert client.vault.get(queued) is None  # it will never run
    with closing(connect(client.db_path)) as conn:
        claimable = conn.execute("SELECT id FROM reviews WHERE status = 'queued'").fetchall()
    assert {row["id"] for row in claimable} == {other_pr, newest}


def test_a_head_force_pushed_back_to_an_earlier_commit_is_reviewed_again(
    client: Client,
) -> None:
    first = client.request("POST", "/v1/reviews", REQUEST).json()["id"]
    second = client.request("POST", "/v1/reviews", {**REQUEST, "head_sha": "c" * 40}).json()["id"]

    back = client.request("POST", "/v1/reviews", REQUEST)

    assert back.status_code == 202
    assert back.json()["id"] not in (first, second)
    assert (
        client.request("GET", f"/v1/reviews/{second}").json()["superseded_by"] == back.json()["id"]
    )


def test_the_database_allows_one_live_review_per_commit(client: Client) -> None:
    client.request("POST", "/v1/reviews", REQUEST)

    with closing(connect(client.db_path)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode, event)"
            " VALUES ('octo/shop', 7, ?, ?, 'standard', 'synchronize')",
            ("a" * 40, HEAD),
        )


def test_a_manual_review_request_starts_a_fresh_review_of_the_same_commit(
    client: Client,
) -> None:
    first = client.request("POST", "/v1/reviews", REQUEST).json()["id"]

    manual = client.request("POST", "/v1/reviews", {**REQUEST, "event": "manual"})

    assert manual.status_code == 202
    assert manual.json()["id"] != first


def test_a_secret_written_into_a_finding_is_masked_in_the_poll(client: Client) -> None:
    # CI logs what it polls; a model may have copied a token into a finding (issue #15).
    leaked = "ghp_" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2"
    review_id = client.request("POST", "/v1/reviews", REQUEST).json()["id"]
    with closing(connect(client.db_path)) as conn:
        conn.execute(
            "INSERT INTO findings (review_id, fingerprint, primary_dimension, severity,"
            " file, line_start, line_end, title, body, evidence, basis)"
            " VALUES (?, 'f1', 'security', 'important', 'a.py', 3, 4, ?, ?, 'e', ?)",
            (
                review_id,
                f"token {leaked} committed",
                f"It is {leaked}.",
                json.dumps([{"kind": "path", "explanation": leaked}]),
            ),
        )

    [found] = client.request("GET", f"/v1/reviews/{review_id}").json()["findings"]

    assert leaked not in json.dumps(found)
    assert found["title"] == "token *** committed" and found["basis"][0]["explanation"] == "***"
