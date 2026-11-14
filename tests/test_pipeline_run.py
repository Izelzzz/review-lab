import json
import re
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from gitrepo import Upstream, git
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from scripted import ScriptedModel, call

from review_agents.config import ModelEndpoint, Settings
from review_agents.context.git import Snapshot
from review_agents.context.pack import GENERAL, GENERAL_TASK
from review_agents.credentials import TokenVault
from review_agents.db import connect, migrate
from review_agents.mirror import MirrorStore
from review_agents.models import BudgetExceededError
from review_agents.pipeline.agent import SUBMIT_TOOL
from review_agents.pipeline.merge import MERGE_TOOL
from review_agents.pipeline.nits import NIT_TOOL
from review_agents.pipeline.run import ReviewSuperseded, run_review, wait_for_summaries
from review_agents.pipeline.verify import VERIFY_TOOL
from review_agents.repo_config import MAX_REVIEW_MINUTES
from review_agents.walkthrough import MARKER, WALKTHROUGH_TASK, WALKTHROUGH_TOOL
from review_agents.worker import claim_next, recover_interrupted

DIMENSION = re.compile(r"^name: (\w+)$", re.MULTILINE)
VERIFIER = "You are now a verifier"
TITLE = re.compile(r"^title: (.+)$", re.MULTILINE)
PLACE = re.compile(r"^location: (.+):(\d+)-(\d+)$", re.MULTILINE)


def holds(severity: str | None = None) -> dict[str, Any]:
    return {"holds": True, "severity": severity}


def does_not_hold() -> dict[str, Any]:
    return {"holds": False}


def _verdict(answer: dict[str, Any], finding_text: str) -> AIMessage:
    place = PLACE.search(finding_text)
    assert place is not None
    args: dict[str, Any] = {"holds": answer["holds"], "reason": "Read the lines."}
    if answer.get("severity"):
        args["severity"] = answer["severity"]
    if answer["holds"]:
        args["basis"] = [
            {
                "kind": "path",
                "places": [
                    {
                        "file": place.group(1),
                        "line_start": int(place.group(2)),
                        "line_end": int(place.group(3)),
                        "shows": "the amount is not clamped",
                    }
                ],
                "explanation": "The amount is returned as requested.",
            }
        ]
    call = {"name": VERIFY_TOOL, "args": args, "id": "v"}
    return AIMessage("", tool_calls=[call])


class ByDimension(BaseChatModel):
    """Answers each dimension agent from its own script, safe to share across threads."""

    scripts: dict[str, list[Any]]
    delay: dict[str, float] = Field(default_factory=dict)
    # A verifier's answer by the finding's title: holds() unless given; an exception
    # is raised. Delays by title too.
    verdicts: dict[str, Any] = Field(default_factory=dict)
    # The walkthrough agent's replies, one per model call; prose when there are none.
    walkthrough: list[Any] = Field(default_factory=list)
    seen: list[str] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)  # titles of the findings verified
    order: list[str] = Field(default_factory=list)  # every call: a dimension or "verify:<title>"
    sent: list[list[BaseMessage]] = Field(default_factory=list)
    lock: Any = Field(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "by-dimension"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ByDimension":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if any(VERIFIER in str(m.content) for m in messages):
            # The finding's block; a short review may have told the verifier to answer now.
            text = next(
                str(m.content) for m in messages if '<untrusted source="finding">' in str(m.content)
            )
            match = TITLE.search(text)
            assert match is not None
            title = match.group(1)
            time.sleep(self.delay.get(f"verify:{title}", 0.0))
            with self.lock:
                self.checked.append(title)
                self.order.append(f"verify:{title}")
            with self.lock:  # a list is a verifier's turns, one per model call
                answer = self.verdicts.get(title, holds())
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
            if callable(answer):  # something happening while the verifier works
                answer = answer()
            if isinstance(answer, AIMessage):
                return ChatResult(generations=[ChatGeneration(message=answer)])
            if isinstance(answer, Exception):
                raise answer
            return ChatResult(generations=[ChatGeneration(message=_verdict(answer, text))])

        # The latest task in the conversation says who is asked: an agent may carry on
        # from the walkthrough's conversation (strategy walkthrough_first).
        def last(found: Any) -> int:
            return max((i for i, m in enumerate(messages) if found(str(m.content))), default=-1)

        wrote = last(lambda text: WALKTHROUGH_TASK in text)
        asked = last(lambda text: bool(DIMENSION.search(text)) or GENERAL_TASK in text)
        if wrote > asked:
            with self.lock:
                self.order.append("walkthrough")
                said = self.walkthrough.pop(0) if self.walkthrough else AIMessage("It clamps.")
            if isinstance(said, Exception):
                raise said
            return ChatResult(generations=[ChatGeneration(message=said)])
        task = str(messages[asked].content) if asked >= 0 else ""
        match = DIMENSION.search(task)
        if match is None:  # a module summary for the repository layer
            return ChatResult(generations=[ChatGeneration(message=AIMessage("A module."))])
        dimension = GENERAL if GENERAL_TASK in task else match.group(1)
        time.sleep(self.delay.get(dimension, 0.0))
        with self.lock:
            self.seen.append(dimension)
            self.order.append(dimension)
            self.sent.append(list(messages))
            script = self.scripts.get(dimension) or [submit()]
            reply = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


class JudgeAll(BaseChatModel):
    """A merge judge that puts every candidate into one group."""

    kind: bool = False
    text: bool = False  # answers in prose instead of calling the tool
    listings: list[str] = Field(default_factory=list)  # the candidates it was shown
    unknown_id: bool = False  # names a candidate that does not exist

    @property
    def _llm_type(self) -> str:
        return "judge-all"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "JudgeAll":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.text:
            return ChatResult(generations=[ChatGeneration(message=AIMessage("they look alike"))])
        self.listings.append(str(messages[-1].content))
        ids = [int(i) for i in re.findall(r"^#(\d+) ", str(messages[-1].content), re.MULTILINE)]
        ids += [999] if self.unknown_id else []
        args = (
            {"same_problem": [[i] for i in ids], "same_kind": [ids]}
            if self.kind
            else {"same_problem": [ids], "same_kind": []}
        )
        call = {"name": MERGE_TOOL, "args": args, "id": "m"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])


def finding(dimension: str, **changes: Any) -> dict[str, Any]:
    return {
        "title": f"{dimension}: refunds can exceed the paid amount",
        "severity": "important",
        "dimension": dimension,
        "file": "payments/refund.py",
        "line_start": 1,
        "line_end": 2,
        "basis": [
            {
                "kind": "spec",
                "source": "pr",
                "quote": "Never refund more than paid.",
                "places": [
                    {"file": "payments/refund.py", "line_start": 1, "line_end": 2, "shows": "s"}
                ],
                "explanation": "refund(order, 10_000) on an order paid 100",
            }
        ],
        "why": "Customers are refunded more than they paid.",
        "suggestion": "Clamp to order.paid.",
        **changes,
    }


def submit(*findings: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": SUBMIT_TOOL, "args": {"findings": list(findings)}, "id": "s"}],
    )


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        self.upstream = Upstream(tmp_path, "octo/shop")
        self.base = self.upstream.commit(
            {"payments/refund.py": "def refund(order, cents):\n    return cents\n"}, "init"
        )
        git(self.upstream.work, "checkout", "-q", "-b", "pr")
        self.head = self.upstream.commit(
            {"payments/refund.py": "def refund(order, cents):\n    return min(cents, 1)\n"},
            "clamp",
            branch="pr",
        )
        git(self.upstream.work, "checkout", "-q", "main")
        # GitHub keeps every PR's head under refs/pull/<number>/head.
        git(self.upstream.bare, "update-ref", "refs/pull/7/head", self.head)
        self.settings = Settings(
            data_dir=tmp_path / "data",
            finder_model=ModelEndpoint.model_validate(
                {"base_url": "http://127.0.0.1:9/v1", "model": "finder"}
            ),
        )
        self.settings.data_dir.mkdir()
        with closing(connect(self.settings.db_path)) as conn:
            migrate(conn)
        self.store = MirrorStore(self.settings.repos_dir, remote_base=str(self.upstream.base))
        self.vault = TokenVault()

    def queue(self, config: str | None = None, head: str | None = None) -> int:
        if config is not None:
            self.upstream.commit({".github/review-agents.yml": config}, "config")
        with closing(connect(self.settings.db_path)) as conn:
            row = conn.execute(
                "INSERT INTO reviews (repo, pr, base_sha, head_sha, mode, default_branch, title,"
                " body, event, status, started_at) VALUES ('octo/shop', 7, ?, ?, 'standard',"
                " 'main', 'Clamp refunds', 'Never refund more than paid.', 'opened', 'running',"
                " strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) RETURNING id",
                (self.base, head or self.head),
            ).fetchone()
        return int(row["id"])

    def run(self, review_id: int, model: ByDimension, **options: Any) -> None:
        with closing(connect(self.settings.db_path)) as conn:
            run_review(
                conn,
                review_id,
                settings=self.settings,
                vault=self.vault,
                store=self.store,
                model_for=lambda timeout: model,
                **options,
            )

    def findings(self, review_id: int) -> list[dict[str, Any]]:
        with closing(connect(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT primary_dimension, dimensions, severity, reported_severity, file, title,"
                " body, suggestion, fingerprint, votes, locations, verdict, basis FROM findings"
                " WHERE review_id = ? ORDER BY id",
                (review_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def steps(self, review_id: int) -> dict[str, tuple[str, str | None]]:
        with closing(connect(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT node, status, error FROM steps WHERE review_id = ? ORDER BY id",
                (review_id,),
            ).fetchall()
        return {row["node"]: (row["status"], row["error"]) for row in rows}

    def notes(self, review_id: int) -> dict[str, str | None]:
        with closing(connect(self.settings.db_path)) as conn:
            rows = conn.execute(
                "SELECT node, note FROM steps WHERE review_id = ? ORDER BY id", (review_id,)
            ).fetchall()
        return {row["node"]: row["note"] for row in rows}

    def review(self, review_id: int) -> dict[str, Any]:
        with closing(connect(self.settings.db_path)) as conn:
            return dict(conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone())


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[Setup]:
    yield Setup(tmp_path)


ALL = ("spec_drift", "edge_cases", "security", "maintainability", "verifiability")


def test_every_enabled_dimension_runs_and_its_findings_are_written(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert sorted(set(model.seen)) == sorted(ALL)
    [written] = setup.findings(review_id)
    assert written["primary_dimension"] == "edge_cases"
    assert json.loads(written["dimensions"]) == ["edge_cases"]
    assert written["body"] == "Customers are refunded more than they paid."
    assert written["suggestion"] == "Clamp to order.paid."
    assert len(written["fingerprint"]) == 64
    steps = setup.steps(review_id)
    assert all(steps[f"dimension.{d}"] == ("ok", None) for d in ALL)
    assert steps["context.pr_layer"][0] == "ok"


def test_only_the_dimensions_the_repository_enables_run(setup: Setup) -> None:
    config = "dimensions:\n  spec_drift: false\n  maintainability: false\n  verifiability: false\n"
    review_id = setup.queue(config)
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert sorted(set(model.seen)) == ["edge_cases", "security"]


def test_one_failing_dimension_leaves_the_others_and_says_why(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "security": [RuntimeError("model service unavailable")],
            "edge_cases": [submit(finding("edge_cases"))],
        }
    )

    setup.run(review_id, model)

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    status, error = setup.steps(review_id)["dimension.security"]
    assert status == "error" and "model service unavailable" in (error or "")


def test_a_dimension_still_running_at_its_deadline_is_left_behind(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={"edge_cases": [submit(finding("edge_cases"))]}, delay={"security": 3.0}
    )

    started = time.monotonic()
    setup.run(review_id, model, review_seconds=2.5, after_submit_seconds=1.0)

    assert time.monotonic() - started < 2.8  # it did not wait for the slow dimension
    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    status, error = setup.steps(review_id)["dimension.security"]
    assert status == "error" and "latest submit time" in (error or "")
    assert setup.review(review_id)["limit_hit"] == "time"


def test_a_spent_token_budget_is_recorded_on_the_review(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={d: [BudgetExceededError("budget used up")] for d in ALL})

    setup.run(review_id, model)

    assert setup.review(review_id)["limit_hit"] == "tokens"


def test_a_superseded_review_stops_and_writes_nothing(setup: Setup) -> None:
    review_id = setup.queue()
    newer = setup.queue(head="c" * 40)
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET superseded_by = ? WHERE id = ?", (newer, review_id))
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    with pytest.raises(ReviewSuperseded, match=f"review {newer}"):
        setup.run(review_id, model)

    assert setup.findings(review_id) == []
    assert setup.review(review_id)["report"] is None
    assert model.seen == []


def test_a_review_whose_token_was_lost_fails_before_any_work(setup: Setup) -> None:
    review_id = setup.queue()
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET has_token = 1 WHERE id = ?", (review_id,))

    with pytest.raises(RuntimeError, match="token"):
        setup.run(review_id, ByDimension(scripts={}))


def test_findings_are_written_all_at_once_or_not_at_all(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    review_id = setup.queue()
    # Two different problems, so the merge keeps both and there are two writes.
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases", line_start=1, line_end=1))],
            "security": [submit(finding("security", line_start=2, line_end=2))],
        }
    )
    from review_agents.pipeline import run as run_module

    original = run_module._insert_finding
    calls = []

    def fail_on_second(*args: Any) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        original(*args)

    monkeypatch.setattr(run_module, "_insert_finding", fail_on_second)

    with pytest.raises(RuntimeError, match="disk full"):
        setup.run(review_id, model)

    assert setup.findings(review_id) == []


def test_after_a_restart_a_review_without_a_token_is_queued_once_more(setup: Setup) -> None:
    review_id = setup.queue()
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET attempts = 1 WHERE id = ?", (review_id,))
        recover_interrupted(conn)
        assert conn.execute("SELECT status FROM reviews").fetchone()[0] == "queued"
        assert claim_next(conn) == review_id  # a second attempt
        recover_interrupted(conn)
        row = conn.execute("SELECT status, error FROM reviews").fetchone()
    assert row["status"] == "failed" and "restart" in row["error"]


def test_a_review_shorter_than_the_after_submit_reserve_still_reviews(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    # 60 seconds in all, less than the 600-second reserve on its own.
    setup.run(review_id, model, review_seconds=60.0)

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]


def test_missing_module_summaries_are_written_in_the_background(setup: Setup) -> None:
    review_id = setup.queue()
    slow_summaries = threading.Event()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    original = model._generate

    def summaries_wait(messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        if not any(DIMENSION.search(str(m.content)) for m in messages):
            slow_summaries.wait(5)  # a module summary: held until the review is done
        return original(messages, *args, **kwargs)

    object.__setattr__(model, "_generate", summaries_wait)

    setup.run(review_id, model)  # returns although the summaries are still waiting

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    slow_summaries.set()
    wait_for_summaries()
    with closing(connect(setup.settings.db_path)) as conn:
        written = conn.execute(
            "SELECT count(*) FROM context_cache WHERE cache_key LIKE 'module:%'"
        ).fetchone()[0]
    assert written >= 1


def test_findings_in_an_unreadable_file_do_not_share_a_fingerprint(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As for a file over the size limit or a binary one: there is no text to fingerprint.
    monkeypatch.setattr(Snapshot, "read_text", lambda self, path: None)
    review_id = setup.queue()
    first = finding("edge_cases", line_start=1, line_end=1, title="one")
    second = finding("edge_cases", line_start=2, line_end=2, title="two")
    model = ByDimension(scripts={"edge_cases": [submit(first, second)]})

    setup.run(review_id, model)

    prints = {f["fingerprint"] for f in setup.findings(review_id)}
    assert len(prints) == 2


def test_stored_findings_and_done_are_one_commit_so_a_restart_cannot_repeat_them(
    setup: Setup,
) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)
    # The process dies before the worker's own bookkeeping; the next one recovers.
    with closing(connect(setup.settings.db_path)) as conn:
        recover_interrupted(conn)
        assert claim_next(conn) is None  # nothing to run again

    assert setup.review(review_id)["status"] == "done"
    assert len(setup.findings(review_id)) == 1


def test_a_single_enabled_dimension_completes_the_review(setup: Setup) -> None:
    others = "\n".join(f"  {d}: false" for d in ALL if d != "security")
    review_id = setup.queue(f"dimensions:\n{others}\n")
    model = ByDimension(scripts={"security": [submit(finding("security"))]})

    setup.run(review_id, model)

    assert set(model.seen) == {"security"}
    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["security"]
    assert setup.steps(review_id)["config"] == ("ok", None)


def test_quick_mode_runs_only_edge_cases_and_security(setup: Setup) -> None:
    review_id = setup.queue("mode: quick\n")
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert set(model.seen) == {"edge_cases", "security"}


def test_quick_mode_falls_back_to_an_enabled_dimension(setup: Setup) -> None:
    others = "\n".join(f"  {d}: false" for d in ALL if d != "maintainability")
    review_id = setup.queue(f"mode: quick\ndimensions:\n{others}\n")
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert set(model.seen) == {"maintainability"}


def test_deep_mode_runs_each_dimension_three_times_and_counts_votes(setup: Setup) -> None:
    others = "\n".join(f"  {d}: false" for d in ALL if d != "edge_cases")
    review_id = setup.queue(f"mode: deep\ndimensions:\n{others}\n")
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert model.seen.count("edge_cases") == 3
    [stored] = setup.findings(review_id)  # the same problem, reported in all three runs
    assert stored["votes"] == 3
    steps = setup.steps(review_id)
    assert {"dimension.edge_cases.run1", "dimension.edge_cases.run3"} <= set(steps)


def test_the_judge_merges_one_problem_two_dimensions_found(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases"))],
            "security": [submit(finding("security", line_end=1, why="much more to say"))],
        }
    )
    judge = JudgeAll()

    setup.run(review_id, model, judge_for=lambda timeout: judge)

    [stored] = setup.findings(review_id)
    assert json.loads(stored["dimensions"]) == ["edge_cases", "security"]
    assert setup.steps(review_id)["merge"] == ("ok", None)


@pytest.mark.parametrize(
    ("judge", "reason"),
    [
        (JudgeAll(text=True), "judge failed: ValueError"),
        (JudgeAll(unknown_id=True), "judge answer rejected"),
    ],
)
def test_a_merge_that_falls_back_to_the_safe_merge_is_recorded(
    setup: Setup, judge: JudgeAll, reason: str
) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases"))],
            "security": [submit(finding("security"))],
        }
    )

    setup.run(review_id, model, judge_for=lambda timeout: judge)

    status, error = setup.steps(review_id)["merge"]
    assert status == "error"
    assert error is not None and error.startswith(reason)
    assert len(setup.findings(review_id)) == 2  # the review still completes
    assert setup.review(review_id)["status"] == "done"


def test_without_a_judge_the_safe_merge_is_no_fallback(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases"))],
            "security": [submit(finding("security"))],
        }
    )

    setup.run(review_id, model)

    assert setup.steps(review_id)["merge"] == ("ok", None)


def test_one_kind_of_problem_is_stored_once_with_its_other_places(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={
            "edge_cases": [
                submit(
                    finding("edge_cases", line_start=1, line_end=1),
                    finding("edge_cases", line_start=2, line_end=2, title="again"),
                )
            ]
        }
    )

    setup.run(review_id, model, judge_for=lambda timeout: JudgeAll(kind=True))

    [stored] = setup.findings(review_id)
    assert json.loads(stored["locations"]) == [["payments/refund.py", 2, 2]]


def test_the_judge_is_not_asked_when_there_is_nothing_to_merge(setup: Setup) -> None:
    asked: list[float] = []

    def judge_for(timeout: float) -> JudgeAll:
        asked.append(timeout)
        return JudgeAll()

    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    setup.run(review_id, model, judge_for=judge_for)

    assert asked == []


def test_the_judge_gets_the_time_left_to_the_deadline(setup: Setup) -> None:
    asked: list[float] = []

    def judge_for(timeout: float) -> JudgeAll:
        asked.append(timeout)
        return JudgeAll()

    review_id = setup.queue()
    model = ByDimension(scripts={d: [submit(finding(d))] for d in ("edge_cases", "security")})
    setup.run(review_id, model, judge_for=judge_for, review_seconds=30.0)

    assert len(asked) == 1 and 0 < asked[0] <= 30.0  # the time to the review's deadline


def test_the_mirror_sync_gets_the_time_the_review_has(setup: Setup) -> None:
    # The repository's own, shorter limit is in the mirror, so the sync is bounded by the
    # longest review there can be (decision D-028).
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    sync = setup.store.sync
    deadlines: list[float] = []

    def recording_sync(*args: Any, deadline: float, **kwargs: Any) -> Path:
        deadlines.append(deadline)
        return sync(*args, deadline=deadline, **kwargs)

    setup.store.sync = recording_sync  # type: ignore[method-assign,assignment]
    setup.run(review_id, model, clock=lambda: 1000.0)

    assert deadlines == [1000.0 + MAX_REVIEW_MINUTES * 60]


def test_the_review_keeps_tool_results_within_the_server_limits(setup: Setup) -> None:
    setup.settings = setup.settings.model_copy(update={"tool_result_tokens": 10})
    review_id = setup.queue()
    read = AIMessage(
        "",
        tool_calls=[{"name": "read_file", "args": {"path": "payments/refund.py"}, "id": "r1"}],
    )
    model = ByDimension(scripts={"edge_cases": [read, submit()]})

    setup.run(review_id, model)

    results = [str(m.content) for sent in model.sent for m in sent if m.type == "tool"]
    assert results and all("Shortened" in r for r in results)


# Verification (issue #10; decisions D-027, D-031, D-032).


def verify_steps(setup: Setup, review_id: int) -> dict[str, tuple[str, str | None]]:
    return {k: v for k, v in setup.steps(review_id).items() if k.startswith("verify.")}


def test_a_finding_is_published_only_once_a_verifier_says_it_holds(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert model.checked == ["edge_cases: refunds can exceed the paid amount"]
    [written] = setup.findings(review_id)
    verdict = json.loads(written["verdict"])
    assert verdict["holds"] is True and verdict["reason"] == "Read the lines."
    assert verdict["basis"][0]["places"][0]["file"] == "payments/refund.py"
    assert verdict["basis"][0]["check"]["status"] == "ok"
    assert written["severity"] == written["reported_severity"] == "important"
    assert list(verify_steps(setup, review_id).values()) == [("ok", None)]


def test_a_finding_that_does_not_hold_is_not_published_and_the_step_says_why(
    setup: Setup,
) -> None:
    review_id = setup.queue()
    wrong = finding("security", title="security: nothing at all", line_start=2, line_end=2)
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases", line_start=1, line_end=1))],
            "security": [submit(wrong)],
        },
        verdicts={"security: nothing at all": does_not_hold()},
    )

    setup.run(review_id, model)

    assert [f["title"] for f in setup.findings(review_id)] == [
        "edge_cases: refunds can exceed the paid amount"
    ]
    notes = [n for k, n in setup.notes(review_id).items() if k.startswith("verify.")]
    assert any(n and "security: nothing at all" in n and "does not hold" in n for n in notes)


def test_a_verifier_may_change_the_level_and_the_reported_one_is_kept(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={"edge_cases": [submit(finding("edge_cases"))]},
        verdicts={"edge_cases: refunds can exceed the paid amount": holds("blocking")},
    )

    setup.run(review_id, model)

    [written] = setup.findings(review_id)
    assert (written["severity"], written["reported_severity"]) == ("blocking", "important")


def test_nits_are_published_without_verification(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases", severity="nit"))]})

    setup.run(review_id, model)

    assert model.checked == []
    [written] = setup.findings(review_id)
    assert written["severity"] == "nit" and written["verdict"] is None


def test_quick_mode_publishes_without_verification(setup: Setup) -> None:
    review_id = setup.queue("mode: quick\n")
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert model.checked == []
    assert len(setup.findings(review_id)) == 1


def test_findings_are_verified_while_slower_dimensions_still_work(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={"edge_cases": [submit(finding("edge_cases"))]}, delay={"security": 1.0}
    )

    setup.run(review_id, model)

    order = model.order
    assert order.index("verify:edge_cases: refunds can exceed the paid amount") < order.index(
        "security"
    )


def test_one_dimension_s_repeated_runs_are_verified_once(setup: Setup) -> None:
    review_id = setup.queue("mode: deep\n")
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    assert model.checked == ["edge_cases: refunds can exceed the paid amount"]
    [written] = setup.findings(review_id)
    assert written["votes"] == 3


def test_a_failed_verification_leaves_the_others_and_is_not_published(setup: Setup) -> None:
    review_id = setup.queue()
    broken = finding("security", title="security: unverifiable", line_start=2, line_end=2)
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases", line_start=1, line_end=1))],
            "security": [submit(broken)],
        },
        verdicts={"security: unverifiable": RuntimeError("verifier unavailable")},
    )

    setup.run(review_id, model)

    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    errors = [e for status, e in verify_steps(setup, review_id).values() if status == "error"]
    assert len(errors) == 1 and "verifier unavailable" in (errors[0] or "")


def test_a_review_none_of_whose_findings_could_be_verified_fails(setup: Setup) -> None:
    review_id = setup.queue()
    title = "edge_cases: refunds can exceed the paid amount"
    model = ByDimension(
        scripts={"edge_cases": [submit(finding("edge_cases"))]},
        verdicts={title: RuntimeError("verifier unavailable")},
    )

    with pytest.raises(RuntimeError, match="no finding could be verified"):
        setup.run(review_id, model)

    assert setup.findings(review_id) == []


def test_a_review_with_nothing_to_verify_does_not_fail(setup: Setup) -> None:
    review_id = setup.queue()

    setup.run(review_id, ByDimension(scripts={}))

    assert setup.findings(review_id) == [] and setup.review(review_id)["status"] == "done"


def test_verifiers_use_the_verifier_model_when_there_is_one(setup: Setup) -> None:
    review_id = setup.queue()
    finder = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})
    verifier = ByDimension(scripts={})

    setup.run(review_id, finder, verifier_for=lambda left: verifier)

    assert finder.checked == []
    assert verifier.checked == ["edge_cases: refunds can exceed the paid amount"]


def test_a_finding_still_unverified_at_the_end_is_not_published(setup: Setup) -> None:
    review_id = setup.queue()
    slow = finding("security", title="security: slow to check", line_start=2, line_end=2)
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases", line_start=1, line_end=1))],
            "security": [submit(slow)],
        },
        delay={"verify:security: slow to check": 4.0},
    )

    started = time.monotonic()
    setup.run(review_id, model, review_seconds=2.5, after_submit_seconds=1.5)

    assert time.monotonic() - started < 3.5  # the review did not wait for it
    assert [f["primary_dimension"] for f in setup.findings(review_id)] == ["edge_cases"]
    errors = [e for status, e in verify_steps(setup, review_id).values() if status == "error"]
    assert errors and "deadline" in (errors[0] or "")


def test_only_verified_findings_are_merged(setup: Setup) -> None:
    review_id = setup.queue()
    judge = JudgeAll()
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases"))],
            "security": [submit(finding("security"))],
            "spec_drift": [submit(finding("spec_drift", title="spec_drift: not real"))],
        },
        verdicts={"spec_drift: not real": does_not_hold()},
    )

    setup.run(review_id, model, judge_for=lambda left: judge)

    assert "spec_drift: not real" not in "\n".join(judge.listings)
    [merged] = setup.findings(review_id)
    assert sorted(json.loads(merged["dimensions"])) == ["edge_cases", "security"]


def test_a_blocking_finding_is_verified_ahead_of_robustness_ones_already_started(
    setup: Setup,
) -> None:
    # One model call at a time; two robustness checks take many turns each.
    setup.settings = setup.settings.model_copy(update={"model_concurrency": 1})
    review_id = setup.queue()
    looks = [
        AIMessage("", tool_calls=[{"name": "list_dir", "args": {}, "id": f"l{n}"}])
        for n in range(6)
    ]
    slow_a = finding(
        "edge_cases", title="edge_cases: a", severity="robustness", line_start=1, line_end=1
    )
    slow_b = finding(
        "edge_cases", title="edge_cases: b", severity="robustness", line_start=2, line_end=2
    )
    urgent = finding("security", title="security: urgent", severity="blocking")
    model = ByDimension(
        scripts={"edge_cases": [submit(slow_a, slow_b)], "security": [submit(urgent)]},
        delay={"security": 0.3, "verify:edge_cases: a": 0.1, "verify:edge_cases: b": 0.1},
        verdicts={"edge_cases: a": [*looks, holds()], "edge_cases: b": [*list(looks), holds()]},
    )

    setup.run(review_id, model)

    order = model.order
    # Its first call comes as soon as a call ends, not once a robustness check is done.
    assert order.index("verify:security: urgent") < len(order) - 1 - order[::-1].index(
        "verify:edge_cases: a"
    )
    assert order.index("verify:security: urgent") < len(order) - 1 - order[::-1].index(
        "verify:edge_cases: b"
    )


def test_a_review_superseded_while_verifying_is_superseded_not_failed(setup: Setup) -> None:
    review_id = setup.queue()
    newer = setup.queue(head="c" * 40)
    title = "edge_cases: refunds can exceed the paid amount"

    def pushed_then_failed() -> Any:
        with closing(connect(setup.settings.db_path)) as conn:
            conn.execute("UPDATE reviews SET superseded_by = ? WHERE id = ?", (newer, review_id))
        raise RuntimeError("verifier unavailable")

    model = ByDimension(
        scripts={"edge_cases": [submit(finding("edge_cases"))]},
        verdicts={title: pushed_then_failed},
    )

    with pytest.raises(ReviewSuperseded):
        setup.run(review_id, model)


# Bases (issue #68; decision D-036).


def test_a_published_finding_keeps_its_bases_and_how_they_checked_out(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    [written] = setup.findings(review_id)
    [stored] = json.loads(written["basis"])
    assert stored["kind"] == "spec" and stored["quote"] == "Never refund more than paid."
    assert stored["check"] == {"status": "ok", "problem": None}


def test_quick_mode_does_not_publish_an_important_finding_whose_basis_did_not_check_out(
    setup: Setup,
) -> None:
    # Quick mode has no verifier to look again (decision D-036).
    review_id = setup.queue("mode: quick\n")
    misquoted = finding("edge_cases", title="edge_cases: misquoted", line_start=2, line_end=2)
    misquoted["basis"] = [dict(misquoted["basis"][0], quote="Refund twice.")]
    robust = finding("security", severity="robustness", title="security: robust", basis=[])
    model = ByDimension(
        scripts={
            "edge_cases": [submit(finding("edge_cases", line_start=1, line_end=1), misquoted)],
            "security": [submit(robust)],
        }
    )

    setup.run(review_id, model)

    titles = [f["title"] for f in setup.findings(review_id)]
    assert "edge_cases: misquoted" not in titles
    assert {"edge_cases: refunds can exceed the paid amount", "security: robust"} <= set(titles)
    notes = [n for n in setup.notes(review_id).values() if n]
    assert any("edge_cases: misquoted" in n and "the quote is not in" in n for n in notes)


def report_of(setup: Setup, review_id: int) -> dict[str, Any] | None:
    stored = setup.review(review_id)["report"]
    return None if stored is None else dict(json.loads(stored))


def test_the_report_is_written_with_the_findings(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None
    assert report["counts"] == {
        "blocking": 0,
        "important": 1,
        "robustness": 0,
        "nit": 0,
        "complete": True,
    }
    # The finding spans lines 1-2; only line 2 changed.
    [comment] = report["comments"]
    assert (comment["path"], comment["start_line"], comment["line"]) == (
        "payments/refund.py",
        None,
        2,
    )
    [written] = setup.findings(review_id)
    assert f"<!-- review-agents:finding {written['fingerprint']} -->" in comment["body"]
    # The verifier's basis, not the dimension agent's.
    assert "The amount is returned as requested." in comment["body"]
    assert "refund(order, 10_000) on an order paid 100" not in comment["body"]
    summary = report["summary"]
    assert summary.startswith("<!-- review-agents:summary -->")
    assert "审了：" in summary and "1 个改动文件" in summary and f"审查 #{review_id}" in summary


def test_a_dimension_that_failed_makes_the_report_incomplete(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"security": [RuntimeError("model service unavailable")]})

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None and report["counts"]["complete"] is False
    unfinished = report["summary"].split("### 未完成的部分")[1]
    assert "安全维度" in unfinished and "model service unavailable" in unfinished


def test_a_quick_review_s_report_says_it_was_not_verified(setup: Setup) -> None:
    review_id = setup.queue("mode: quick\nlanguage: en\n")
    model = ByDimension(scripts={"edge_cases": [submit(finding("edge_cases"))]})

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None
    assert "were not confirmed by a verifier" in report["summary"]
    assert "Quick mode" in report["summary"]


NIT_DIMENSIONS = ("edge_cases", "security", "maintainability", "verifiability")


def eight_nits() -> dict[str, list[Any]]:
    """Two nits a dimension, on lines 1 and 2: no two of them are merged as one problem."""
    return {
        d: [
            submit(
                *(
                    finding(d, severity="nit", title=f"{d} nit {n}", line_start=n, line_end=n)
                    for n in (1, 2)
                )
            )
        ]
        for d in NIT_DIMENSIONS
    }


def test_nits_past_the_limit_are_merged_by_the_strong_model(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts=eight_nits())
    groups = [
        {"ids": ids, "title": title, "why": "w", "suggestion": "s"}
        for ids, title in (([1, 2, 3, 4], "a"), ([5, 6, 7, 8], "b"))
    ]
    strong = ScriptedModel(script=[call(NIT_TOOL, {"groups": groups})])

    setup.run(review_id, model, verifier_for=lambda left: strong)

    report = report_of(setup, review_id)
    assert report is not None and report["counts"]["nit"] == 2
    assert "Simplified Chinese" in str(strong.sent[0][0].content)
    assert setup.steps(review_id)["report.nits"] == ("ok", None)
    assert len(setup.findings(review_id)) == 8  # the findings themselves stay as found


def test_a_failed_nit_merge_shows_the_first_nits_and_says_why(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts=eight_nits())

    setup.run(review_id, model)  # the finder answers in prose instead of grouping

    report = report_of(setup, review_id)
    assert report is not None and report["counts"]["nit"] == 5
    status, error = setup.steps(review_id)["report.nits"]
    assert status == "error" and error is not None and NIT_TOOL in error


def test_the_agents_are_told_the_repository_s_language(setup: Setup) -> None:
    review_id = setup.queue("language: en\n")
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert all("in English" in str(sent[-1].content) for sent in model.sent)


def walkthrough_answer() -> AIMessage:
    card = {"headline": "Refunds are clamped", "points": ["one", "two"]}
    args = {
        "summary": "Refunds never exceed what was paid.",
        "why": card,
        "what": card,
        "impact": card,
        "problems": [
            {"title": "Over-refunds", "explanation": "More was refunded than paid."},
            {"title": "No cap", "explanation": "Nothing compared it with the payment."},
        ],
        "changes": [{"title": "Clamp", "body": "`payments/refund.py:2` clamps the amount."}],
    }
    return AIMessage("", tool_calls=[{"name": WALKTHROUGH_TOOL, "args": args, "id": "w"}])


def test_the_walkthrough_is_written_with_the_report(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={}, walkthrough=[walkthrough_answer()])

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None
    text = report["walkthrough"]
    assert text.startswith(MARKER) and "Refunds never exceed what was paid." in text
    assert "PR #7 · 1 个文件 · **+1 −1** · 1 次提交" in text
    assert setup.steps(review_id)["walkthrough"] == ("ok", None)


def test_the_walkthrough_is_published_when_a_dimension_fails(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(
        scripts={"security": [RuntimeError("model service unavailable")]},
        walkthrough=[walkthrough_answer()],
    )

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None and report["counts"]["complete"] is False
    assert "Refunds never exceed what was paid." in report["walkthrough"]


def test_a_walkthrough_not_handed_in_is_replaced_by_why(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={})  # the walkthrough agent answers in prose, twice

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None and report["walkthrough"].startswith(MARKER)
    assert "改动说明没能生成" in report["walkthrough"]
    status, error = setup.steps(review_id)["walkthrough"]
    assert status == "error" and error is not None


# --- strategies (issue #83, D-038) ---------------------------------------------


def with_strategy(setup: Setup, strategy: str) -> None:
    setup.settings = setup.settings.model_copy(update={"strategy": strategy})


def test_the_general_strategy_runs_one_agent_for_every_dimension(setup: Setup) -> None:
    with_strategy(setup, "general")
    review_id = setup.queue()
    model = ByDimension(
        scripts={GENERAL: [submit(finding("edge_cases"))]}, walkthrough=[walkthrough_answer()]
    )

    setup.run(review_id, model)

    assert model.seen == [GENERAL]
    [written] = setup.findings(review_id)
    assert written["primary_dimension"] == "edge_cases"  # the dimension it tagged
    steps = setup.steps(review_id)
    assert steps["dimension.general"] == ("ok", None)
    assert steps["walkthrough"][0] == "ok"  # written alongside, as before
    report = json.loads(setup.review(review_id)["report"])
    assert "规格漂移" in report["summary"] and "可验证性" in report["summary"]  # all reviewed


def test_walkthrough_first_finds_problems_after_the_walkthrough(setup: Setup) -> None:
    with_strategy(setup, "walkthrough_first")
    review_id = setup.queue()
    model = ByDimension(
        scripts={GENERAL: [submit(finding("security"))]},
        walkthrough=[walkthrough_answer()],
    )

    setup.run(review_id, model)

    assert model.order[:2] == ["walkthrough", GENERAL]
    [sent] = model.sent  # the agent's one turn: the walkthrough's conversation, then its task
    assert WALKTHROUGH_TASK in str(sent[2].content) and GENERAL_TASK in str(sent[-1].content)
    [written] = setup.findings(review_id)
    assert written["primary_dimension"] == "security"
    report = json.loads(setup.review(review_id)["report"])
    assert report["walkthrough"].startswith(MARKER)


def test_walkthrough_first_still_finds_problems_when_the_walkthrough_fails(setup: Setup) -> None:
    with_strategy(setup, "walkthrough_first")
    review_id = setup.queue()
    model = ByDimension(
        scripts={GENERAL: [submit(finding("security"))]},
        walkthrough=[RuntimeError("model service down")],
    )

    setup.run(review_id, model)

    assert len(setup.findings(review_id)) == 1
    [sent] = model.sent
    assert not any(WALKTHROUGH_TASK in str(m.content) for m in sent)  # a fresh opening
    status, error = setup.steps(review_id)["walkthrough"]
    assert status == "error" and error is not None


# --- rounds (issue #12, D-039) ---------------------------------------------------

EARLIER_SUMMARY = (
    "<!-- review-agents:summary -->\n## 审查结论：未发现阻断问题\n\n- 🟡 earlier finding\n\n"
    '<!-- review-agents:counts {"blocking": 0, "important": 1, "complete": true} -->\n'
)


def earlier(setup: Setup, round_: int, status: str = "done", superseded: bool = False) -> int:
    """A review of the same PR that came before, as the worker left it."""
    earlier.count += 1  # type: ignore[attr-defined]
    review_id = setup.queue(head=f"{earlier.count:040x}")  # type: ignore[attr-defined]
    report = {"summary": EARLIER_SUMMARY, "comments": [], "counts": {"complete": True}}
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute(
            "UPDATE reviews SET status = ?, round = ?, report = ?, superseded_by = ?,"
            " finished_at = 'x' WHERE id = ?",
            (status, round_, json.dumps(report), review_id if superseded else None, review_id),
        )
    return review_id


earlier.count = 0  # type: ignore[attr-defined]


def test_a_review_takes_the_round_after_the_last_finished_one(setup: Setup) -> None:
    earlier(setup, 1)
    earlier(setup, 2)
    earlier(setup, 3, status="failed")  # a failed or superseded review is not a round
    earlier(setup, 3, superseded=True)
    review_id = setup.queue()

    setup.run(review_id, ByDimension(scripts={}))

    assert setup.review(review_id)["round"] == 3


def test_a_pr_s_first_review_is_round_one(setup: Setup) -> None:
    review_id = setup.queue()

    setup.run(review_id, ByDimension(scripts={}))

    assert setup.review(review_id)["round"] == 1


def test_past_ten_rounds_a_push_is_not_reviewed(setup: Setup) -> None:
    earlier(setup, 10)
    review_id = setup.queue()
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    assert model.seen == [] and model.order == []  # no model was asked anything
    review = setup.review(review_id)
    assert review["status"] == "done" and review["round"] == 10
    report = json.loads(review["report"])
    assert report["skipped"] == {"reason": "round_limit", "round": 11, "limit": 10}
    assert report["counts"] is None and report["comments"] == []
    summary = report["summary"]
    assert summary.startswith("<!-- review-agents:summary -->")
    assert "10 轮" in summary and "/review" in summary
    assert "earlier finding" in summary  # the last round's findings stay up
    assert "review-agents:counts" not in summary
    assert setup.findings(review_id) == []


def test_a_repository_may_review_fewer_rounds(setup: Setup) -> None:
    earlier(setup, 2)
    review_id = setup.queue("limits:\n  full_review_rounds: 2\n")

    setup.run(review_id, ByDimension(scripts={}))

    assert json.loads(setup.review(review_id)["report"])["skipped"]["limit"] == 2


def test_a_review_command_reviews_past_the_limit(setup: Setup) -> None:
    earlier(setup, 10)
    review_id = setup.queue()
    with closing(connect(setup.settings.db_path)) as conn:
        conn.execute("UPDATE reviews SET event = 'manual' WHERE id = ?", (review_id,))
    model = ByDimension(scripts={})

    setup.run(review_id, model)

    review = setup.review(review_id)
    assert review["round"] == 11 and "skipped" not in json.loads(review["report"])
    assert sorted(set(model.seen)) == sorted(ALL)


# --- secrets (issue #15) ---------------------------------------------------------

LEAKED = "ghp_" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7m6L5k4J3i2"


def test_an_error_carrying_a_secret_is_stored_masked(setup: Setup) -> None:
    review_id = setup.queue()
    model = ByDimension(scripts={"security": [RuntimeError(f"401 for token {LEAKED}")]})

    setup.run(review_id, model)

    status, error = setup.steps(review_id)["dimension.security"]
    assert status == "error" and error is not None
    assert LEAKED not in error and "***" in error


def test_a_secret_quoted_in_a_finding_is_masked_in_the_report(setup: Setup) -> None:
    review_id = setup.queue()
    leak = finding("security", why=f"The token {LEAKED} is committed.")
    model = ByDimension(scripts={"security": [submit(leak)]})

    setup.run(review_id, model)

    report = report_of(setup, review_id)
    assert report is not None
    text = json.dumps(report, ensure_ascii=False)
    assert LEAKED not in text and "***" in text
