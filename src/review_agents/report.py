"""The report of one review: a summary comment and inline comments (report design §5-§7).

The service writes the report and CI only posts it (decision D-034). The summary groups
the findings by level; the 🔴 and 🟡 on the PR's changed lines are also inline comments,
the others are listed in a section of their own (D-033). Two hidden markers let CI find
the summary and read the counts without parsing it, and each inline comment carries its
finding's fingerprint, so CI posts it once (D-033). Only bases that checked out are
shown (D-036).

The words of the report are the configured language's; what the agents wrote is shown as
they wrote it, with anything that could pass for a marker or a mention defused.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from review_agents.pipeline.agent import Severity
from review_agents.pipeline.basis import Basis, Place, checked_out
from review_agents.repo_config import Language

SUMMARY_MARKER = "<!-- review-agents:summary -->"
LEVELS: tuple[Severity, ...] = ("blocking", "important", "robustness", "nit")
INLINE: tuple[Severity, ...] = ("blocking", "important")
SKIPPED_SHOWN = 10  # skipped files listed by name; past that, only how many


@dataclass(frozen=True)
class Item:
    """A published finding, as the report shows it."""

    fingerprint: str
    severity: Severity
    title: str
    file: str
    line_start: int
    line_end: int
    why: str
    suggestion: str
    dimensions: list[str]
    preexisting: bool
    # Other places the same problem appears.
    locations: list[tuple[str, int, int]]
    # The verifier's bases for a verified finding, the dimension agent's otherwise.
    bases: list[Basis]


@dataclass(frozen=True)
class MissedStep:
    """A dimension agent that did not hand in. `group` and `run` are set when there are
    several of them (a large PR in groups, Deep mode's runs)."""

    dimension: str
    group: int | None
    run: int | None
    reason: str


@dataclass(frozen=True)
class Unverified:
    title: str
    severity: Severity
    reason: str


@dataclass(frozen=True)
class Scope:
    mode: str
    dimensions: list[str]
    runs: int  # how many times each dimension ran
    files: int  # changed files the agents were shown
    groups: int
    skipped: list[str]  # changed files that match a skip rule
    binary: list[str]
    finder_verifies: bool  # no verifier model is configured
    # Quick mode's 🔴 / 🟡 not published because no basis checked out (D-036).
    withheld: int = 0
    # Options in the repository's config that no longer do anything (D-037).
    retired: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Stats:
    tokens: int
    seconds: float
    review_id: int


@dataclass(frozen=True)
class Comment:
    path: str
    start_line: int | None  # set when the comment spans several lines
    line: int
    body: str


@dataclass(frozen=True)
class Report:
    summary: str
    comments: list[Comment] = field(default_factory=list)
    counts: dict[str, int | bool] = field(default_factory=dict)
    # The first comment, the walkthrough (D-035); set once it is written.
    walkthrough: str | None = None
    # Why the push was not reviewed (D-039): CI shows the required check as skipped.
    skipped: dict[str, object] | None = None

    def as_json(self) -> dict[str, object]:
        out: dict[str, object] = {
            "summary": self.summary,
            "comments": [asdict(c) for c in self.comments],
            "counts": self.counts or None,
            "walkthrough": self.walkthrough,
        }
        if self.skipped is not None:
            out["skipped"] = self.skipped
        return out


_TEXT: dict[Language, dict[str, str]] = {
    "zh": {
        "blocking": "🔴 阻断",
        "important": "🟡 重要",
        "robustness": "🔵 健壮性建议",
        "nit": "⚪ 细节",
        "must_fix": "审查结论：需要先处理 {n} 条 🔴",
        "no_blocking": "审查结论：未发现阻断问题",
        "round_limit": "审查结论：这次推送没有审查",
        "round_limit_why": "这个 PR 已经自动审查了 {limit} 轮，到了上限，之后的推送不再自动审查。"
        "改了很多轮还在改，往往说明方案本身值得重新审视。需要时在 PR 下评论 `/review`，"
        "会再完整审查一次。",
        "round_limit_last": "以下是第 {round} 轮（最后一次审查）的结论，没有包含之后的改动：",
        "incomplete": "审查结论：审查未完成，以下只是已完成部分的结果",
        "count": "（{n}）",
        "see_inline": "见行间评论",
        "see_below": "见下方「不在本次改动行上的发现」",
        "found_by": "{names}都发现了",
        "join": "、",
        "sep": "；",
        "preexisting": "（已存在）",
        "also_at": "同类问题还在：{places}",
        "why": "原因：",
        "suggestion": "建议：",
        "because": "为什么：",
        "basis": "依据 · {kind}",
        "spec": "规格",
        "contract": "契约",
        "convention": "约定",
        "path": "路径",
        "pr": "PR 描述",
        "off_lines": "不在本次改动行上的发现",
        "off_lines_note": "以下问题出在这次 PR 没有改动的代码上，GitHub 不允许在这些行发行间"
        "评论，所以列在这里。",
        "unfinished": "未完成的部分",
        "dimension": "{name}维度",
        "group": "第 {n} 组",
        "run": "第 {n} 次",
        "unverified": "未能验证：{title}（报为 {level}）",
        "scope": "本次审查的范围",
        "reviewed": "审了：{names}（各 {runs} 次）· {files} 个改动文件",
        "in_groups": " · 分 {n} 组",
        "not_reviewed": "没审：{what}",
        "nothing_left": "改动文件都看了，各维度都交了卷",
        "skipped": "命中跳过规则的 {n} 个文件（{names}）",
        "skipped_more": "命中跳过规则的 {n} 个文件（{names} 等）",
        "binary": "二进制文件 {names}",
        "missed": "{n} 个维度没交卷",
        "not_verified": "{n} 条发现没验完",
        "withheld": "{n} 条 🔴 / 🟡 的依据没核对通过，没有发布",
        "proves": "能证明：发布的每条 🔴 / 🟡 都有程序核对过的依据，并经验证审查员重新读"
        "代码确认；引用的规则原文确实在所写的位置。",
        "proves_quick": "能证明：引用的规则原文确实在所写的位置。Quick 模式下发布的 🔴 / 🟡 "
        "只经过程序核对依据，没有经过验证审查员复核。",
        "cannot": "不能证明：没有运行代码和测试；没有发现问题不代表没有问题；规格只来自 PR "
        "描述、关联的 issue 和配置的规格文档；没验完的发现没有发布，不代表它们不成立。",
        "finder_verifies": "验证用的是找问题模型（没有配置验证模型）。",
        "retired": "仓库配置里的 {names} 已不再使用：是否拦合入，请在 GitHub 的分支保护里把这个检查"
        "设成必须通过。",
        "this_review": "本次审查",
        "stats": "{mode} 模式 · {tokens} token · {time} · 审查 #{id}",
        "minutes": "{m} 分 {s} 秒",
        "seconds": "{s} 秒",
        "spec_drift": "规格漂移",
        "edge_cases": "边界问题",
        "security": "安全",
        "maintainability": "可维护性",
        "verifiability": "可验证性",
        "general": "通用审查员",
        "did not hand in its findings by the latest submit time": "没在最晚交卷时间前交卷",
        "not started before the latest submit time": "到最晚交卷时间还没开始",
        "the review's deadline passed before it was verified": "到审查截止仍未验完",
    },
    "en": {
        "blocking": "🔴 Blocking",
        "important": "🟡 Important",
        "robustness": "🔵 Robustness",
        "nit": "⚪ Nits",
        "must_fix": "Review: {n} 🔴 to address first",
        "no_blocking": "Review: No blocking problems found",
        "round_limit": "Review: this push was not reviewed",
        "round_limit_why": "This PR has been reviewed {limit} times, the limit, so later pushes"
        " are not reviewed automatically. Many rounds of changes often mean the approach"
        " itself deserves a second look. Comment `/review` on the PR for one more full review.",
        "round_limit_last": "What round {round}, the last review, found; later changes are"
        " not in it:",
        "incomplete": "Review: incomplete; what follows covers only the parts that finished",
        "count": " ({n})",
        "see_inline": "see the inline comment",
        "see_below": 'see "Not on the changed lines" below',
        "found_by": "found by {names}",
        "join": ", ",
        "sep": "; ",
        "preexisting": " (pre-existing)",
        "also_at": "The same problem is also at: {places}",
        "why": "Why: ",
        "suggestion": "Suggestion: ",
        "because": "Because: ",
        "basis": "Basis · {kind}",
        "spec": "spec",
        "contract": "contract",
        "convention": "convention",
        "path": "path",
        "pr": "PR description",
        "off_lines": "Not on the changed lines",
        "off_lines_note": "These problems are in code this PR does not change. GitHub does not"
        " allow inline comments on those lines, so they are listed here.",
        "unfinished": "What did not finish",
        "dimension": "{name}",
        "group": "group {n}",
        "run": "run {n}",
        "unverified": "Not verified: {title} (reported {level})",
        "scope": "What this review covered",
        "reviewed": "Reviewed: {names} ({runs} each) · {files} changed files",
        "in_groups": " · in {n} groups",
        "not_reviewed": "Not reviewed: {what}",
        "nothing_left": "every changed file was read and every dimension handed in",
        "skipped": "{n} files matching a skip rule ({names})",
        "skipped_more": "{n} files matching a skip rule ({names} and more)",
        "binary": "binary files {names}",
        "missed": "{n} dimensions that did not hand in",
        "not_verified": "{n} findings not verified",
        "withheld": "{n} 🔴 / 🟡 not published: no basis checked out",
        "proves": "Shows: every published 🔴 / 🟡 has a basis the program checked, and a"
        " verifier read the code again to confirm it; every quoted rule is where it says.",
        "proves_quick": "Shows: every quoted rule is where it says. In Quick mode the published"
        " 🔴 / 🟡 had their bases checked by the program but were not confirmed by a verifier.",
        "cannot": "Does not show: no code or tests were run; finding no problem does not mean"
        " there is none; specs come only from the PR description, its linked issues and the"
        " configured spec documents; findings not verified in time were not published, which"
        " does not mean they are wrong.",
        "finder_verifies": "The finder model verified (no verifier model is configured).",
        "retired": "{names} in the repository's config is no longer used: to block merging,"
        " make this check required in GitHub's branch protection.",
        "this_review": "This review",
        "stats": "{mode} mode · {tokens} tokens · {time} · review #{id}",
        "minutes": "{m} min {s} s",
        "seconds": "{s} s",
        "spec_drift": "spec drift",
        "edge_cases": "edge cases",
        "security": "security",
        "maintainability": "maintainability",
        "verifiability": "verifiability",
        "general": "general reviewer",
        "did not hand in its findings by the latest submit time": "did not hand in by the"
        " latest submit time",
        "not started before the latest submit time": "had not started by the latest submit time",
        "the review's deadline passed before it was verified": "not verified by the review's"
        " deadline",
    },
}

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(patch: str) -> list[tuple[int, int]]:
    """The head-side lines a file's patch adds, as ranges: the lines GitHub takes inline
    comments on, whatever context the patch was made with."""
    ranges: list[tuple[int, int]] = []
    line: int | None = None
    for text in patch.splitlines():
        if text.startswith("diff --git"):
            line = None
        elif match := _HUNK.match(text):
            line = int(match.group(1))
        elif line is None:
            continue
        elif text.startswith("+"):
            if ranges and ranges[-1][1] == line - 1:
                ranges[-1] = (ranges[-1][0], line)
            else:
                ranges.append((line, line))
            line += 1
        elif text.startswith(" "):
            line += 1
    return ranges


def is_binary(patch: str) -> bool:
    return any(
        (t.startswith("Binary files ") and t.endswith(" differ")) or t == "GIT binary patch"
        for t in patch.splitlines()
    )


def defused(text: str) -> str:
    """Model text cannot open a hidden marker or mention a GitHub user. A `-->` alone
    opens nothing, and Mermaid draws its arrows with it, so it stays."""
    return text.replace("<!--", "<!​--").replace("@", "@​")


def _line(text: str) -> str:
    return defused(" ".join(text.split()))


def _code(text: str) -> str:
    """Text in a code span it cannot leave: one line, no backticks, no marker. A mention
    in a code span does not notify anyone, so `@` stays as it is."""
    text = " ".join(text.split()).replace("`", "'")
    return "`" + text.replace("<!--", "<!​--").replace("-->", "--​>") + "`"


def _where(file: str, start: int, end: int) -> str:
    return _code(f"{file}:{start}" if start == end else f"{file}:{start}-{end}")


class _Writer:
    def __init__(self, language: Language) -> None:
        self.t = _TEXT[language]
        self.colon = "：" if language == "zh" else ": "

    def aside(self, text: str) -> str:
        """Text in brackets after what it is about."""
        return f"（{text}）" if self.colon == "：" else f" ({text})"

    def name(self, dimension: str) -> str:
        return self.t.get(dimension, dimension)

    def names(self, dimensions: Sequence[str]) -> str:
        return self.t["join"].join(self.name(d) for d in dimensions)

    def reason(self, reason: str) -> str:
        return self.t.get(reason) or _line(reason)

    def place(self, place: Place) -> str:
        where = _where(place.file, place.line_start, place.line_end)
        return f"{where}{self.colon}{_line(place.shows)}"

    def basis(self, basis: Basis, indent: str) -> list[str]:
        t = self.t
        head = f"{indent}- " + t["basis"].format(kind=t[basis.kind])
        if basis.kind == "path":
            lines = [head]
            lines += [
                f"{indent}  {n}. {self.place(p)}" for n, p in enumerate(basis.places, start=1)
            ]
        else:
            source = basis.source or ""
            shown = t["pr"] if source.strip().lower() in {"pr", "pr description"} else _code(source)
            lines = [f"{head} {shown}", f"{indent}  > {_line(basis.quote or '')}"]
            lines += [f"{indent}  - {self.place(p)}" for p in basis.places]
        lines.append(f"{indent}  {t['because']}{_line(basis.explanation)}")
        return lines

    def bases(self, item: Item, indent: str) -> list[str]:
        return [line for b in item.bases if checked_out(b) for line in self.basis(b, indent)]

    def notes(self, item: Item) -> list[str]:
        """Who found it and where else it is, when there is anything to say."""
        t = self.t
        out = []
        if len(item.dimensions) > 1:
            out.append(t["found_by"].format(names=self.names(item.dimensions)))
        if item.locations:
            places = t["join"].join(_where(*loc) for loc in item.locations)
            out.append(t["also_at"].format(places=places))
        return out

    def details(self, item: Item) -> list[str]:
        """What a finding listed in the summary says under its title."""
        t = self.t
        lines = [f"  {t['why']}{_line(item.why)}", *self.bases(item, "  ")]
        if item.suggestion:
            lines.append(f"  {t['suggestion']}{_line(item.suggestion)}")
        return lines + [f"  {note}" for note in self.notes(item)]

    def title(self, item: Item) -> str:
        return _line(item.title) + (self.t["preexisting"] if item.preexisting else "")

    def heading(self, key: str, n: int | None = None) -> str:
        count = "" if n is None else self.t["count"].format(n=n)
        return f"### {self.t[key]}{count}"


def build_report(
    items: Sequence[Item],
    *,
    changed: dict[str, list[tuple[int, int]]],
    missed: Sequence[MissedStep],
    unverified: Sequence[Unverified],
    scope: Scope,
    stats: Stats,
    language: Language,
    nits_per_report: int,
) -> Report:
    """The report of one finished review.

    `changed` gives each file's changed lines; `items` are in the order to show them.
    Nits past `nits_per_report` are left out (D-034).
    """
    w = _Writer(language)
    t = w.t
    nits = [i for i in items if i.severity == "nit"][:nits_per_report]
    shown = [i for i in items if i.severity != "nit"] + nits
    inline: list[tuple[Item, int | None, int]] = []
    off: list[Item] = []
    for item in shown:
        if item.severity not in INLINE:
            continue
        anchor = _anchor(item, changed)
        if anchor is None:
            off.append(item)
        else:
            inline.append((item, *anchor))
    complete = not missed and not any(u.severity in INLINE for u in unverified)
    counts: dict[str, int | bool] = {
        level: sum(1 for i in shown if i.severity == level) for level in LEVELS
    }
    counts["complete"] = complete

    if counts["blocking"]:
        conclusion = t["must_fix"].format(n=counts["blocking"])
    elif complete:
        conclusion = t["no_blocking"]
    else:
        conclusion = t["incomplete"]
    lines = [SUMMARY_MARKER, f"## {conclusion}"]

    for level in LEVELS:
        group = [i for i in shown if i.severity == level]
        if not group:
            continue
        lines += ["", w.heading(level, len(group))]
        for item in group:
            entry = f"- {w.title(item)} — {_where(item.file, item.line_start, item.line_end)}"
            if level in INLINE:
                # Its details are in its inline comment, or below when it has none.
                see = t["see_below"] if any(item is o for o in off) else t["see_inline"]
                lines.append(entry + w.aside(t["sep"].join([see, *w.notes(item)])))
            else:
                lines += [entry, *w.details(item)]

    if off:
        lines += ["", w.heading("off_lines", len(off)), f"> {t['off_lines_note']}"]
        for item in off:
            where = _where(item.file, item.line_start, item.line_end)
            lines.append(f"- {t[item.severity].split(' ')[0]} {w.title(item)} — {where}")
            lines += w.details(item)

    if missed or unverified:
        lines += ["", w.heading("unfinished")]
        for step in missed:
            which = [t["group"].format(n=step.group)] if step.group else []
            which += [t["run"].format(n=step.run)] if step.run else []
            name = t["dimension"].format(name=w.name(step.dimension))
            if which:
                name += w.aside(t["join"].join(which))
            lines.append(f"- {name}{w.colon}{w.reason(step.reason)}")
        for finding in unverified:
            mark = t[finding.severity].split(" ")[0]
            head = t["unverified"].format(title=_line(finding.title), level=mark)
            lines.append(f"- {head}{w.colon}{w.reason(finding.reason)}")

    lines += ["", w.heading("scope"), *_scope(w, scope, missed, unverified)]
    lines += ["", w.heading("this_review"), _stats(w, scope, stats)]
    lines += ["", f"<!-- review-agents:counts {json.dumps(counts)} -->"]

    comments = [Comment(item.file, start, end, _comment(w, item)) for item, start, end in inline]
    return Report("\n".join(lines) + "\n", comments, counts)


def _anchor(item: Item, changed: dict[str, list[tuple[int, int]]]) -> tuple[int | None, int] | None:
    """The finding's lines within one stretch of changed lines: GitHub takes a comment's
    first and last line only inside one stretch of the diff."""
    for first, last in changed.get(item.file, []):
        start, end = max(first, item.line_start), min(last, item.line_end)
        if start <= end:
            return (start if start < end else None), end
    return None


def _comment(w: _Writer, item: Item) -> str:
    t = w.t
    parts = [f"**{t[item.severity]} · {w.title(item)}**", defused(item.why.strip())]
    bases = w.bases(item, "")
    if bases:
        parts.append("\n".join(bases))
    if item.suggestion:
        parts.append(f"**{t['suggestion']}**{defused(item.suggestion.strip())}")
    notes = w.notes(item)
    if notes:
        parts.append(t["sep"].join(notes))
    parts.append(f"<!-- review-agents:finding {item.fingerprint} -->")
    return "\n\n".join(p for p in parts if p) + "\n"


def _scope(
    w: _Writer, scope: Scope, missed: Sequence[MissedStep], unverified: Sequence[Unverified]
) -> list[str]:
    t = w.t
    runs: object = scope.runs
    if w.colon != "：":
        runs = "once" if scope.runs == 1 else f"{scope.runs} runs"
    reviewed = t["reviewed"].format(names=w.names(scope.dimensions), runs=runs, files=scope.files)
    if scope.groups > 1:
        reviewed += t["in_groups"].format(n=scope.groups)
    left_out = []
    if scope.skipped:
        names = t["join"].join(_code(p) for p in scope.skipped[:SKIPPED_SHOWN])
        key = "skipped_more" if len(scope.skipped) > SKIPPED_SHOWN else "skipped"
        left_out.append(t[key].format(n=len(scope.skipped), names=names))
    if scope.binary:
        left_out.append(t["binary"].format(names=t["join"].join(_code(p) for p in scope.binary)))
    if missed:
        left_out.append(t["missed"].format(n=len({m.dimension for m in missed})))
    if unverified:
        left_out.append(t["not_verified"].format(n=len(unverified)))
    if scope.withheld:
        left_out.append(t["withheld"].format(n=scope.withheld))
    lines = [
        f"- {reviewed}",
        f"- {t['not_reviewed'].format(what=t['sep'].join(left_out) or t['nothing_left'])}",
        f"- {t['proves_quick'] if scope.mode == 'quick' else t['proves']}",
        f"- {t['cannot']}",
    ]
    if scope.finder_verifies and scope.mode != "quick":
        lines.append(f"- {t['finder_verifies']}")
    if scope.retired:
        names = t["join"].join(_code(name) for name in scope.retired)
        lines.append(f"- {t['retired'].format(names=names)}")
    return lines


def _stats(w: _Writer, scope: Scope, stats: Stats) -> str:
    t = w.t
    minutes, seconds = divmod(round(stats.seconds), 60)
    took = t["minutes"].format(m=minutes, s=seconds) if minutes else t["seconds"].format(s=seconds)
    return t["stats"].format(
        mode=scope.mode.capitalize(), tokens=f"{stats.tokens:,}", time=took, id=stats.review_id
    )


_COUNTS_LINE = re.compile(r"\n?<!-- review-agents:counts .*?-->\n?")


def round_limit_report(
    round_: int, limit: int, last: str | None, last_round: int, language: Language
) -> Report:
    """The summary of a push past the round limit (D-039): not reviewed, and why.

    The last review's findings stay up beneath, without its counts line: CI takes a
    summary without one for one with no conclusion of its own.
    """
    t = _TEXT[language]
    parts = [SUMMARY_MARKER, f"## {t['round_limit']}", "", t["round_limit_why"].format(limit=limit)]
    if last:
        body = _COUNTS_LINE.sub("\n", last.replace(SUMMARY_MARKER, "", 1)).strip()
        parts += ["", t["round_limit_last"].format(round=last_round), "", body]
    return Report(
        summary="\n".join(parts) + "\n",
        skipped={"reason": "round_limit", "round": round_, "limit": limit},
    )
