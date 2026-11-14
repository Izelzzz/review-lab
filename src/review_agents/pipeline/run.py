"""One review from the queue to its stored findings (pipeline design §2, §6, §7).

The steps run in plain Python (decision D-023): sync the mirror, read the config from
the default branch, build the context, find and verify, merge, write the findings.
Between steps the review checks that a newer push has not superseded it (D-010).

Finding and verifying have no time limits of their own (decision D-031). The dimension
agents run side by side; as each hands in, its findings are verified at once, one
verifier per finding, while the others still work. The review's model calls are shared
out by priority: verifying a 🔴, then a 🟡, then a dimension agent's next turn, then a 🔵.
The agents hand in at the latest `after_submit_seconds` before the end; what is still
unverified at the end is not published. Only verified findings, and nits, which are not
verified, are merged and written.

A dimension or a verification that fails or runs out of time ends alone and is recorded
in `steps`; only a failure to get the code, a superseded review, a failed write, or no
finding that needed verifying getting a verdict ends the review.
"""

import hashlib
import json
import logging
import math
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from review_agents.config import Settings
from review_agents.context.cache import repo_layer_for_review
from review_agents.context.git import Snapshot
from review_agents.context.modules import Summarize, module_summaries, module_tokens
from review_agents.context.pack import ContextPack, build_packs
from review_agents.context.pr_layer import PrRequest, build_pr_layer, pr_patches
from review_agents.context.summarizer import model_summarizer
from review_agents.credentials import TokenVault
from review_agents.db import connect
from review_agents.mirror import MirrorStore
from review_agents.models import (
    TokenBudget,
    UsageMeter,
    chat_model,
    review_usage,
    summary_usage,
)
from review_agents.pipeline.agent import AgentRun, Finding, Severity, run_dimension
from review_agents.pipeline.basis import Basis, describe, supported
from review_agents.pipeline.conversation import ConversationLimits
from review_agents.pipeline.merge import Candidate, Merged, merge, model_judge, overlaps
from review_agents.pipeline.nits import NIT_MERGE_SECONDS, merge_nits
from review_agents.pipeline.schedule import DIMENSION_TURN, Slots, verify_priority
from review_agents.pipeline.strategy import Plan, plan
from review_agents.pipeline.verify import VERIFIER_TOOL_CALLS, VerificationRun, run_verifier
from review_agents.redact import REDACT
from review_agents.repo_config import MAX_REVIEW_MINUTES, RepoConfig, read_repo_config
from review_agents.report import (
    Item,
    MissedStep,
    Scope,
    Stats,
    Unverified,
    added_lines,
    build_report,
    is_binary,
    round_limit_report,
)
from review_agents.tools import RepoTools
from review_agents.walkthrough import (
    WALKTHROUGH_TOOL_CALLS,
    WalkthroughRun,
    pr_facts,
    render_failed,
    render_walkthrough,
    run_walkthrough,
)

# Builds a model for one call; its argument is the seconds left to that call's deadline.
ModelFor = Callable[[float], BaseChatModel]
log = logging.getLogger(__name__)
_summaries: dict[str, threading.Thread] = {}
_summaries_guard = threading.Lock()

# When one review hits several limits, the report names the most severe.
_LIMIT_ORDER = ("time", "tokens", "tool_calls")


class ReviewSuperseded(RuntimeError):
    """A newer push of the same PR replaced this review; nothing of it is published."""


@dataclass(frozen=True)
class Job:
    group: int
    groups: int
    dimension: str  # the agent's name: a dimension, or GENERAL
    pack: ContextPack
    run: int = 1
    runs: int = 1
    # The dimensions whose rules the agent holds; its own dimension unless given.
    dimensions: tuple[str, ...] = ()
    # It carries on from the walkthrough's conversation (strategy walkthrough_first).
    after_walkthrough: bool = False

    @property
    def node(self) -> str:
        suffix = f".group{self.group}" if self.groups > 1 else ""
        suffix += f".run{self.run}" if self.runs > 1 else ""
        return f"dimension.{self.dimension}{suffix}"


def plan_jobs(planned: Plan, packs: list[ContextPack]) -> list[Job]:
    """Every agent of the review, for every group of changes, every first run first."""
    return [
        Job(
            index,
            len(packs),
            author.name,
            pack,
            run,
            planned.runs,
            author.dimensions,
            # The walkthrough reads the first group; another group's agent opening from
            # its conversation would see the first group's changes instead of its own.
            planned.walkthrough_first and index == 1,
        )
        for run in range(1, planned.runs + 1)
        for index, pack in enumerate(packs, start=1)
        for author in planned.authors
    ]


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _now(conn: sqlite3.Connection) -> str:
    return str(conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0])


def _step(
    conn: sqlite3.Connection,
    review_id: int,
    node: str,
    started: str,
    error: str | None = None,
    note: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO steps (review_id, node, started_at, finished_at, status, error, note)"
        " VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?)",
        # Masked: an error may quote a token or a key (issue #15).
        (review_id, node, started, "error" if error else "ok", REDACT.optional(error), note),
    )


def _last_round(conn: sqlite3.Connection, review: sqlite3.Row) -> sqlite3.Row | None:
    """The PR's latest round: the last earlier review that finished and was not replaced.

    A failed or superseded review is not a round, nor a push past the limit, which keeps
    the round it found.
    """
    row: sqlite3.Row | None = conn.execute(
        "SELECT round, report FROM reviews WHERE repo = ? AND pr = ? AND id < ?"
        " AND status = 'done' AND superseded_by IS NULL"
        " AND json_extract(report, '$.skipped') IS NULL ORDER BY id DESC LIMIT 1",
        (review["repo"], review["pr"], review["id"]),
    ).fetchone()
    return row


def _skip_past_round_limit(
    conn: sqlite3.Connection,
    review_id: int,
    round_: int,
    limit: int,
    last: sqlite3.Row | None,
    config: RepoConfig,
) -> None:
    """Finish a push past the round limit unreviewed; its summary keeps the last findings."""
    previous = json.loads(last["report"]) if last is not None and last["report"] else {}
    report = round_limit_report(
        round_, limit, previous.get("summary"), last["round"] if last else 0, config.language
    )
    _step(conn, review_id, "rounds", _now(conn), note=f"round {round_} is past the limit {limit}")
    conn.execute(
        "UPDATE reviews SET status = 'done', error = NULL, round = ?, report = ?,"
        " finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        # Masked like every report CI gets (issue #15).
        (round_ - 1, json.dumps(REDACT.every_string(report.as_json())), review_id),
    )


def _check_not_superseded(conn: sqlite3.Connection, review_id: int) -> None:
    row = conn.execute("SELECT superseded_by FROM reviews WHERE id = ?", (review_id,)).fetchone()
    if row["superseded_by"] is not None:
        raise ReviewSuperseded(f"superseded by review {row['superseded_by']}")


def fingerprint(snapshot: Snapshot, finding: Finding) -> str:
    """The same problem at the same code, wherever the lines moved (decision D-022)."""
    text = snapshot.read_text(finding.file)
    if text is None:  # too large or binary to read: the line range stands in for the code
        code = f"lines {finding.line_start}-{finding.line_end}"
    else:
        lines = text.splitlines()[finding.line_start - 1 : finding.line_end]
        code = "\n".join(" ".join(line.split()) for line in lines)
    return hashlib.sha256(f"{finding.file}\n{code}".encode()).hexdigest()


@dataclass
class _Check:
    """One verification, and the same dimension's duplicates waiting on its verdict."""

    number: int
    members: list[Candidate]
    pack: ContextPack
    started: str = ""
    run: VerificationRun | None = None

    @property
    def lead(self) -> Candidate:
        return self.members[0]

    def takes(self, candidate: Candidate) -> bool:
        """The plain duplicate rule of the safe merge (D-024): one dimension, every pair of
        lines overlapping, each agent run once. Such a copy needs no verifier of its own."""
        return candidate.agent not in {m.agent for m in self.members} and all(
            m.finding.dimension == candidate.finding.dimension
            and overlaps(m.finding, candidate.finding)
            for m in self.members
        )


@dataclass
class _Published:
    """What is known of each published candidate besides its finding."""

    reported: dict[int, Severity] = field(default_factory=dict)
    verdicts: dict[int, dict[str, object]] = field(default_factory=dict)
    # The verifier's bases of each verified candidate: what the report shows (D-036).
    bases: dict[int, list[Basis]] = field(default_factory=dict)


def _insert_finding(
    conn: sqlite3.Connection, review_id: int, merged: Merged, published: _Published
) -> None:
    finding = merged.finding
    lead = next(c for c in merged.sources if c.finding is finding)
    verdict = published.verdicts.get(lead.id)
    conn.execute(
        "INSERT INTO findings (review_id, fingerprint, primary_dimension, dimensions, severity,"
        " preexisting, file, line_start, line_end, title, body, suggestion, basis, evidence,"
        " votes, locations, reported_severity, verdict)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            review_id,
            lead.fingerprint,
            finding.dimension,
            json.dumps(merged.dimensions),
            finding.severity,
            finding.preexisting,
            finding.file,
            finding.line_start,
            finding.line_end,
            finding.title,
            finding.why,
            finding.suggestion,
            json.dumps([b.model_dump() for b in finding.basis]),
            "\n\n".join(describe(b) for b in finding.basis),  # for versions before bases
            merged.votes,
            json.dumps(merged.locations),
            published.reported.get(lead.id, finding.severity),
            json.dumps(verdict) if verdict is not None else None,
        ),
    )


def _item(merged: Merged, published: _Published) -> Item:
    """A merged finding as the report shows it: with the verifier's bases when it was
    verified, the dimension agent's otherwise (decision D-036)."""
    finding = merged.finding
    lead = next(c for c in merged.sources if c.finding is finding)
    return Item(
        fingerprint=lead.fingerprint,
        severity=finding.severity,
        title=finding.title,
        file=finding.file,
        line_start=finding.line_start,
        line_end=finding.line_end,
        why=finding.why,
        suggestion=finding.suggestion,
        dimensions=merged.dimensions,
        preexisting=finding.preexisting,
        locations=merged.locations,
        bases=published.bases.get(lead.id, finding.basis),
    )


def _tokens_used(conn: sqlite3.Connection, review_id: int) -> int:
    row = conn.execute(
        "SELECT coalesce(sum(input_tokens + output_tokens), 0) FROM usage WHERE review_id = ?",
        (review_id,),
    ).fetchone()
    return int(row[0])


def run_review(
    conn: sqlite3.Connection,
    review_id: int,
    *,
    settings: Settings,
    vault: TokenVault,
    store: MirrorStore,
    model_for: ModelFor,
    judge_for: ModelFor | None = None,
    verifier_for: ModelFor | None = None,
    clock: Callable[[], float] = time.monotonic,
    review_seconds: float | None = None,
    after_submit_seconds: float | None = None,
) -> None:
    """Review one claimed review and store its findings; raise if it cannot be done.

    `judge_for` merges and `verifier_for` verifies; without a verifier model the finder
    model verifies (D-027), and without a judge only the safe merge runs (D-024).
    """
    if settings.finder_model is None:
        raise RuntimeError("no finder model is configured")
    started_at = clock()
    review = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    token = vault.get(review_id)
    if review["has_token"] and token is None:
        raise RuntimeError("its token is no longer held; run the CI job again")
    repo, head = review["repo"], review["head_sha"]
    _check_not_superseded(conn, review_id)

    step = _now(conn)
    # The default branch for the config and repository layer; the PR's own head ref,
    # which GitHub keeps for forks too, for the commits under review.
    refs = [f"refs/heads/{review['default_branch']}", f"refs/pull/{review['pr']}/head"]
    # The repository's own, shorter limit is read from the mirror, so the sync is
    # bounded by the longest review there can be (decision D-028).
    longest = review_seconds if review_seconds is not None else MAX_REVIEW_MINUTES * 60
    git_dir = store.sync(
        repo, refs, [review["base_sha"], head], token, deadline=started_at + longest, clock=clock
    )
    _step(conn, review_id, "mirror.sync", step)
    step = _now(conn)
    loaded = read_repo_config(git_dir, review["default_branch"])
    config = loaded.config
    # An invalid config falls back to the defaults as a whole (D-001); the step says why.
    _step(conn, review_id, "config", step, "; ".join(loaded.errors) or None)
    limits = config.limits
    # D-039: a push is a round; past the repository's limit only `/review` is reviewed.
    last = _last_round(conn, review)
    round_ = (last["round"] if last is not None else 0) + 1
    if round_ > limits.full_review_rounds and review["event"] != "manual":
        _skip_past_round_limit(conn, review_id, round_, limits.full_review_rounds, last, config)
        return
    conn.execute("UPDATE reviews SET round = ? WHERE id = ?", (round_, review_id))
    deadline = started_at + (
        review_seconds if review_seconds is not None else limits.review_minutes * 60
    )
    total = deadline - started_at
    reserve = (
        after_submit_seconds if after_submit_seconds is not None else settings.after_submit_seconds
    )
    # A repository may shorten the review below the reserve; the agents keep two thirds.
    submit_by = deadline - min(reserve, total / 3)
    budget = TokenBudget(limits.token_budget(settings.tokens_per_review))
    skip = config.effective_skip_paths
    window = settings.finder_model.context_tokens

    _check_not_superseded(conn, review_id)
    # The review reads the summaries that exist; missing ones are written in the
    # background, so a repository's first review does not spend its time on them and
    # git stats stand in until they are there (context design §5).
    repo_layer = repo_layer_for_review(
        conn,
        review_id,
        repo,
        git_dir,
        review["default_branch"],
        skip,
        None,
        module_tokens=module_tokens(window),
    )
    if any(m.summary is None for m in repo_layer.modules):
        summarize = model_summarizer(
            model_for(math.inf),  # summaries belong to no review and have no deadline
            UsageMeter(
                TokenBudget(settings.summary_tokens_per_build),
                summary_usage(settings.db_path, repo),
            ),
        )
        summarize_in_background(
            settings.db_path, repo, git_dir, review["default_branch"], skip,
            module_tokens(window), summarize,
        )  # fmt: skip
    step = _now(conn)
    request = PrRequest(
        repo, review["pr"], review["title"], review["body"], review["base_sha"], head
    )
    pr_layer = build_pr_layer(
        conn, request, git_dir, config, token=token, surrounding_code=settings.surrounding_code
    )
    planned = plan(settings.strategy, config)
    packs = build_packs(
        repo_layer,
        pr_layer,
        window,
        settings.output_reserve_tokens,
        config.language,
        # The room kept for the agents' task follows the agents that run (issue #83).
        [(a.name, a.dimensions) for a in planned.authors],
    )
    # For the report: the whole PR's lines, not only this round's, since GitHub takes
    # inline comments on those. Read with the rest of the code, before any model call.
    patches = pr_patches(git_dir, review["base_sha"], head, pr_layer.files)
    facts = pr_facts(git_dir, pr_layer)  # for the walkthrough's header and file table
    _step(conn, review_id, "context.pr_layer", step)
    if len(packs) > 1:
        _step(conn, review_id, f"context.groups.{len(packs)}", step)

    _check_not_superseded(conn, review_id)
    dimensions, repeats = planned.dimensions, planned.runs
    jobs = plan_jobs(planned, packs)
    # Quick mode publishes what it finds unverified (pipeline design §5).
    verifying = config.mode != "quick"
    if verifying and verifier_for is None:
        _step(
            conn,
            review_id,
            "verifier.model",
            _now(conn),
            note="no verifier model: the finder verifies",
        )
    work = _Work(
        conn,
        review_id,
        settings=settings,
        token=token,
        git_dir=git_dir,
        skip=skip,
        snapshot=Snapshot(git_dir, head, tuple(skip)),
        tool_calls=limits.tool_calls_per_agent,
        budget=budget,
        model_for=model_for,
        verifier_for=verifier_for or model_for,
        clock=clock,
    )
    # The walkthrough reads the first group: the whole PR when it is not split.
    runs, checks, accepted, published = work.find_and_verify(
        jobs, packs[0], submit_by=submit_by, deadline=deadline, verifying=verifying
    )
    # A newer push makes this review superseded, whatever else went wrong in it.
    _check_not_superseded(conn, review_id)
    # D-004 as D-027 has it: findings that needed a verifier, none of which got a verdict,
    # mean the verification itself failed; publishing the nits alone would hide that.
    if checks and not any(c.run is not None and c.run.verdict is not None for c in checks):
        first = next((c.run.reason for c in checks if c.run and c.run.reason), "no verdict")
        raise RuntimeError(f"no finding could be verified ({len(checks)} tried): {first}")

    step = _now(conn)
    judge = None
    fallbacks: list[str] = []
    # One candidate has nothing to merge with; the judge is not asked. It has until the
    # review's end; without time left, the safe merge takes its place at once.
    if judge_for is not None and len(accepted) > 1:
        left = deadline - clock()
        if left > 0:
            meter = UsageMeter(budget, review_usage(settings.db_path, review_id, "merge"))
            judge = model_judge(judge_for(left), callbacks=[meter])
        else:
            fallbacks.append("no time left to judge the merge")
    # A judge that fails or answers nonsense leaves the safe merge: the review still
    # completes, and the step says the merge was degraded.
    merged = merge(accepted, judge, fallbacks.append)
    _step(conn, review_id, "merge", step, fallbacks[0] if fallbacks else None)
    hits = [r.limit_hit for r in runs] + [c.run.limit_hit for c in checks if c.run]
    limit_hit = next((lim for lim in _LIMIT_ORDER if lim in hits), None)
    binary = sorted(path for path, patch in patches.items() if is_binary(patch))
    items = [_item(m, published) for m in merged]
    nits = sum(1 for i in items if i.severity == "nit")
    if 0 < limits.nits_per_report < nits:
        # One call to the strong model, outside the review's time (D-034, D-035).
        step = _now(conn)
        meter = UsageMeter(budget, review_usage(settings.db_path, review_id, "nits"))
        strong = verifier_for or model_for
        nit_merge = merge_nits(
            items,
            lambda: strong(NIT_MERGE_SECONDS),
            limit=limits.nits_per_report,
            callbacks=[meter],
            language=config.language,
        )
        items = nit_merge.items
        _step(conn, review_id, "report.nits", step, nit_merge.problem)
    report = build_report(
        items,
        changed={path: added_lines(patch) for path, patch in patches.items()},
        missed=[
            MissedStep(
                job.dimension,
                job.group if job.groups > 1 else None,
                job.run if job.runs > 1 else None,
                run.reason or "incomplete",
            )
            for job, run in zip(jobs, runs, strict=True)
            if run.status != "done"
        ],
        unverified=[
            Unverified(c.lead.finding.title, c.lead.finding.severity, c.run.reason or "incomplete")
            for c in checks
            if c.run is not None and c.run.verdict is None
        ],
        scope=Scope(
            mode=config.mode,
            dimensions=dimensions,
            runs=repeats,
            files=len(patches) - len(binary),
            groups=len(packs),
            skipped=[f.path for f in pr_layer.files if f.skipped],
            binary=binary,
            finder_verifies=verifier_for is None,
            withheld=work.dropped,
            retired=loaded.retired,
        ),
        stats=Stats(_tokens_used(conn, review_id), clock() - started_at, review_id),
        language=config.language,
        nits_per_report=limits.nits_per_report,
    )
    written = work.walkthrough
    walkthrough = (
        render_walkthrough(written.walkthrough, facts, config.language)
        if written is not None and written.walkthrough is not None
        else render_failed((written and written.reason) or "incomplete", config.language)
    )
    # The findings and the review's end state commit together (reliability §5.3): never
    # half a review, and a review with stored findings is never claimed and run again.
    conn.execute("BEGIN IMMEDIATE")
    try:
        for item in merged:
            _insert_finding(conn, review_id, item, published)
        conn.execute(
            "UPDATE reviews SET limit_hit = ?, status = 'done', error = NULL, report = ?,"
            " finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
            # What CI posts: a secret a finding quotes from the code is masked (issue #15).
            (
                limit_hit,
                json.dumps(REDACT.every_string(replace(report, walkthrough=walkthrough).as_json())),
                review_id,
            ),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


class _Work:
    """The dimension agents and the verifiers of one review, sharing its model calls."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        review_id: int,
        *,
        settings: Settings,
        token: str | None,
        git_dir: Path,
        skip: list[str],
        snapshot: Snapshot,
        tool_calls: int,
        budget: TokenBudget,
        model_for: ModelFor,
        verifier_for: ModelFor,
        clock: Callable[[], float],
    ) -> None:
        assert settings.finder_model is not None  # noqa: S101 - checked when the review starts
        self.conn, self.review_id, self.settings = conn, review_id, settings
        self.token, self.git_dir, self.skip, self.snapshot = token, git_dir, skip, snapshot
        self.tool_calls, self.budget, self.clock = tool_calls, budget, clock
        self.model_for, self.verifier_for = model_for, verifier_for
        row = conn.execute("SELECT repo, head_sha FROM reviews WHERE id = ?", (review_id,))
        self.repo, self.head = row.fetchone()
        self.slots = Slots(settings.model_concurrency)
        self.events: queue.Queue[tuple[str, object, Future[Any] | None]] = queue.Queue()
        self.starts: dict[str, str] = {}
        self.candidates = 0  # ids given so far
        self.dropped = 0  # Quick-mode findings left out for want of a basis
        self.walkthrough: WalkthroughRun | None = None

    def _limits(self, endpoint_tokens: int) -> ConversationLimits:
        return ConversationLimits(
            window=endpoint_tokens,
            result_tokens=self.settings.tool_result_tokens,
            shortened_tokens=self.settings.shortened_result_tokens,
            output_reserve=self.settings.output_reserve_tokens,
        )

    def _tools(self) -> RepoTools:
        return RepoTools(self.git_dir, self.repo, self.head, self.skip, token=self.token)

    def find_and_verify(
        self,
        jobs: list[Job],
        walkthrough_pack: ContextPack,
        *,
        submit_by: float,
        deadline: float,
        verifying: bool,
    ) -> tuple[list[AgentRun], list[_Check], list[Candidate], _Published]:
        """Every job side by side, each one's findings verified as soon as it hands in.

        The walkthrough agent works alongside the dimension agents, at their priority and
        by their latest submit time (D-035); its run is left in `self.walkthrough`.

        Waits for the agents until `submit_by` and for the verifiers until `deadline`.
        What is still running then is left behind: every model call it makes is built
        to end, retries included, by its deadline, so its thread ends soon after.
        """
        settings = self.settings
        assert settings.finder_model is not None  # noqa: S101 - checked when the review starts
        finder_limits = self._limits(settings.finder_model.context_tokens)
        verifier = settings.verifier_model or settings.finder_model
        verifier_limits = self._limits(verifier.context_tokens)

        def find(job: Job) -> AgentRun:
            opening = None
            if job.after_walkthrough:
                # Waits for the walkthrough, queued first, at most until the latest submit
                # time; without it the agent opens its own conversation.
                wait([written], timeout=max(0.0, submit_by - self.clock()))
                if written.done() and not written.cancelled() and written.exception() is None:
                    opening = written.result().transcript
            self.starts[job.node] = _utc_now()  # when the job runs, not when it was queued
            meter = UsageMeter(
                self.budget, review_usage(settings.db_path, self.review_id, job.dimension)
            )
            return run_dimension(
                self.model_for,
                job.pack,
                job.dimension,
                self._tools(),
                tool_calls=self.tool_calls,
                callbacks=[meter],
                deadline=submit_by,
                clock=self.clock,
                limits=finder_limits,
                slot=lambda: self.slots.held(DIMENSION_TURN),
                dimensions=job.dimensions or None,
                opening=opening,
            )

        def write() -> WalkthroughRun:
            self.starts["walkthrough"] = _utc_now()
            meter = UsageMeter(
                self.budget, review_usage(settings.db_path, self.review_id, "walkthrough")
            )
            return run_walkthrough(
                self.model_for,
                walkthrough_pack,
                self._tools(),
                tool_calls=min(self.tool_calls, WALKTHROUGH_TOOL_CALLS),
                callbacks=[meter],
                deadline=submit_by,
                clock=self.clock,
                limits=finder_limits,
                slot=lambda: self.slots.held(DIMENSION_TURN),
            )

        def verify(check: _Check) -> None:
            # Each check has a thread of its own and competes for model calls by priority
            # (Slots): a pool would let checks already started hold every thread until
            # they end, and a 🔴 handed in later would wait behind them.
            check.started = _utc_now()
            meter = UsageMeter(
                self.budget, review_usage(settings.db_path, self.review_id, "verify")
            )
            priority = verify_priority(check.lead.finding.severity)
            try:
                run = run_verifier(
                    self.verifier_for,
                    check.pack,
                    check.lead.finding,
                    self._tools(),
                    tool_calls=VERIFIER_TOOL_CALLS,
                    callbacks=[meter],
                    deadline=deadline,
                    clock=self.clock,
                    limits=verifier_limits,
                    slot=lambda: self.slots.held(priority),
                )
            except Exception as exc:  # one verification failing must not fail the review
                run = VerificationRun(
                    check.lead.finding, "incomplete", reason=f"{type(exc).__name__}: {exc}"
                )
            # Only the review's thread stores the result.
            self.events.put(("verified", (check, run), None))

        finders = ThreadPoolExecutor(settings.model_concurrency, thread_name_prefix="find")
        runs: dict[int, AgentRun] = {}  # by the job's place in `jobs`: a Job is not hashable
        checks: list[_Check] = []
        accepted: list[Candidate] = []
        published = _Published()
        unverified = 0

        def done(kind: str, item: object) -> Callable[[Future[Any]], None]:
            return lambda future: self.events.put((kind, item, future))

        try:
            # Queued first, so it is not left waiting for a thread behind the dimensions,
            # and so an agent that carries on from it never waits for a thread it holds.
            written = finders.submit(write)
            written.add_done_callback(done("written", None))
            for index, job in enumerate(jobs):
                finders.submit(find, job).add_done_callback(done("found", index))
            while True:
                now = self.clock()
                writing = self.walkthrough is None
                agents_out = (len(runs) < len(jobs) or writing) and now < submit_by
                checking = unverified > 0 and now < deadline
                if not (agents_out or checking):
                    break
                try:
                    kind, item, future = self.events.get(
                        timeout=max(0.0, (deadline if checking else submit_by) - now)
                    )
                except queue.Empty:
                    continue
                if kind == "written":
                    assert future is not None  # noqa: S101
                    self._written(future)
                    continue
                if kind == "found":
                    assert isinstance(item, int) and future is not None  # noqa: S101
                    job = jobs[item]
                    run = self._agent_run(job, future)
                    runs[item] = run
                    for candidate in self._candidates(job, run):
                        finding = candidate.finding
                        if not verifying and not self._publishable_unverified(candidate):
                            continue
                        if not verifying or finding.severity == "nit":
                            published.reported[candidate.id] = finding.severity
                            accepted.append(candidate)
                            continue
                        same = next(
                            (c for c in checks if c.pack is job.pack and c.takes(candidate)), None
                        )
                        if same is not None:
                            same.members.append(candidate)
                            if same.run is not None and same.run.verdict is not None:
                                self._accept(same, [candidate], accepted, published)
                            continue
                        check = _Check(len(checks) + 1, [candidate], job.pack)
                        checks.append(check)
                        unverified += 1
                        # A daemon: one still running at the deadline is left behind.
                        threading.Thread(
                            target=verify, args=(check,), name=f"verify-{check.number}", daemon=True
                        ).start()
                else:
                    unverified -= 1
                    assert isinstance(item, tuple)  # noqa: S101
                    check, verification = item
                    check.run = verification
                    self._record(check)
                    self._accept(check, check.members, accepted, published)
        finally:
            finders.shutdown(wait=False, cancel_futures=True)
        for index, job in enumerate(jobs):
            if index not in runs:
                started = job.node in self.starts
                run = AgentRun(job.dimension, "incomplete", limit_hit="time")
                run.reason = (
                    "did not hand in its findings by the latest submit time"
                    if started
                    else "not started before the latest submit time"
                )
                runs[index] = run
                _step(
                    self.conn,
                    self.review_id,
                    job.node,
                    self.starts.get(job.node, _now(self.conn)),
                    run.reason,
                )
        if self.walkthrough is None:
            reason = (
                "did not hand in its walkthrough by the latest submit time"
                if "walkthrough" in self.starts
                else "not started before the latest submit time"
            )
            self.walkthrough = WalkthroughRun("incomplete", reason=reason, limit_hit="time")
            begun = self.starts.get("walkthrough", _now(self.conn))
            _step(self.conn, self.review_id, "walkthrough", begun, reason)
        for check in checks:
            if check.run is None:
                check.run = VerificationRun(
                    check.lead.finding,
                    "incomplete",
                    reason="the review's deadline passed before it was verified",
                    limit_hit="time",
                )
                self._record(check)
        return [runs[index] for index in range(len(jobs))], checks, accepted, published

    def _written(self, future: Future[Any]) -> None:
        if future.cancelled():
            return  # recorded once the agents' time is over
        exc = future.exception()
        if exc is not None:
            run = WalkthroughRun("incomplete", reason=f"{type(exc).__name__}: {exc}")
        else:
            run = future.result()
        self.walkthrough = run
        error = None if run.status == "done" else (run.reason or "incomplete")
        started = self.starts.get("walkthrough", _now(self.conn))
        _step(self.conn, self.review_id, "walkthrough", started, error)

    def _agent_run(self, job: Job, future: Future[Any]) -> AgentRun:
        if future.cancelled():
            run = AgentRun(job.dimension, "incomplete", limit_hit="time")
            run.reason = "not started before the latest submit time"
        elif (exc := future.exception()) is not None:
            run = AgentRun(job.dimension, "incomplete", reason=f"{type(exc).__name__}: {exc}")
        else:
            result = future.result()
            assert isinstance(result, AgentRun)  # noqa: S101
            run = result
        error = None if run.status == "done" else (run.reason or "incomplete")
        _step(
            self.conn, self.review_id, job.node, self.starts.get(job.node, _now(self.conn)), error
        )
        return run

    def _publishable_unverified(self, candidate: Candidate) -> bool:
        """Quick mode has no verifier: a 🔴 / 🟡 is published only on a basis that checked out.

        A basis git was too slow to check counts as not checked out: with no verifier to
        look again, an unchecked quote is not trusted. Recorded in `steps` (decision D-036).
        """
        finding = candidate.finding
        if finding.severity not in ("blocking", "important") or supported(finding.basis):
            return True
        problems = [b.check.problem for b in finding.basis if b.check and b.check.problem]
        why = "; ".join(problems) or "no basis given"
        self.dropped += 1
        _step(
            self.conn,
            self.review_id,
            f"basis.{self.dropped}",
            _now(self.conn),
            note=f"{finding.title}: not published, no basis checked out: {why}",
        )
        return False

    def _candidates(self, job: Job, run: AgentRun) -> list[Candidate]:
        made = []
        for item in run.findings:
            self.candidates += 1
            made.append(
                Candidate(
                    self.candidates, item, job.run, fingerprint(self.snapshot, item), job.node
                )
            )
        return made

    def _accept(
        self,
        check: _Check,
        members: list[Candidate],
        accepted: list[Candidate],
        published: _Published,
    ) -> None:
        if check.run is None or check.run.verdict is None or not check.run.verdict.holds:
            return
        verdict = check.run.verdict
        record: dict[str, object] = {
            "holds": True,
            "reason": verdict.reason,
            "basis": [b.model_dump() for b in verdict.basis],
        }
        for member in members:
            finding = member.finding.model_copy(update={"severity": verdict.severity})
            kept = Candidate(member.id, finding, member.run, member.fingerprint, member.agent)
            published.reported[member.id] = member.finding.severity
            published.verdicts[member.id] = record
            published.bases[member.id] = verdict.basis
            accepted.append(kept)

    def _record(self, check: _Check) -> None:
        run = check.run
        assert run is not None  # noqa: S101
        title = check.lead.finding.title
        started = check.started or _now(self.conn)
        node = f"verify.{check.number}"
        if run.verdict is None:
            _step(
                self.conn, self.review_id, node, started, f"{title}: {run.reason or 'incomplete'}"
            )
            return
        verdict, reported = run.verdict, check.lead.finding.severity
        if not verdict.holds:
            note = f"{title}: does not hold: {verdict.reason}"
        elif verdict.severity != reported:
            note = f"{title}: holds at {verdict.severity} (reported {reported})"
        else:
            note = f"{title}: holds"
        _step(self.conn, self.review_id, node, started, note=note)


def summarize_in_background(
    db_path: Path,
    repo: str,
    git_dir: Path,
    default_branch: str,
    skip: list[str],
    tokens: int,
    summarize: Summarize,
) -> None:
    """Write a repository's missing module summaries on a thread of their own.

    One build per repository at a time; a review that finds one running leaves it be.
    The summaries' own budget (`summary_tokens_per_build`) bounds what a build spends.
    """

    def build() -> None:
        try:
            with closing(connect(db_path, create=False)) as conn:
                snapshot = Snapshot.at_branch(git_dir, default_branch, tuple(skip))
                module_summaries(conn, repo, snapshot, summarize, tokens)
        except Exception:
            log.exception("writing module summaries for %s failed", repo)

    with _summaries_guard:
        running = _summaries.get(repo)
        if running is not None and running.is_alive():
            return
        thread = threading.Thread(target=build, name=f"summaries:{repo}", daemon=True)
        _summaries[repo] = thread
        thread.start()


def wait_for_summaries(timeout: float | None = None) -> None:
    """Wait for the background summary builds; for tests and an orderly shutdown."""
    with _summaries_guard:
        threads = list(_summaries.values())
    for thread in threads:
        thread.join(timeout)


def review_handler(
    settings: Settings, vault: TokenVault
) -> Callable[[sqlite3.Connection, int], None]:
    """The worker's handler: the real pipeline, with the configured finder model."""
    if settings.finder_model is None:
        raise RuntimeError("REVIEW_AGENTS_FINDER_MODEL__* must be set to run reviews")
    endpoint = settings.finder_model
    verifier = settings.verifier_model
    store = MirrorStore(settings.repos_dir)

    def handle(conn: sqlite3.Connection, review_id: int) -> None:
        run_review(
            conn,
            review_id,
            settings=settings,
            vault=vault,
            store=store,
            model_for=lambda left: chat_model(endpoint, time_left=left),
            # The merge is judged by the strong model (D-022); without one, only the safe
            # merge runs: one dimension's agent runs agreeing on overlapping lines.
            judge_for=(lambda left: chat_model(verifier, time_left=left)) if verifier else None,
            # Without a verifier model, the finder model verifies (D-027).
            verifier_for=(lambda left: chat_model(verifier, time_left=left)) if verifier else None,
        )

    return handle
