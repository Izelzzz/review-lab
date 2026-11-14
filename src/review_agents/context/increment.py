"""What a later round reviews: the PR's own changes since the last review (D-026).

The increment is the lines changed between the previously reviewed head and the new
one that are also the PR's own changes against the default branch. What a merge or a
rebase brings in from the default branch is not the PR's change, so it is left out. A
place where the author took back something the PR had added counts too: nothing is
added there, but it was the author's change. An empty increment means there is nothing
new to review (an empty commit, a clean merge or rebase); without the previous head
the round reviews the whole PR.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from review_agents.context.pr_layer import GIT_TIMEOUT_SECONDS
from review_agents.mirror import GitError, check_commit_sha, run_git

Ranges = list[tuple[int, int]]
_FILE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Increment:
    previous_head: str
    # Head-side lines to review, by path; a place taken back is one line, where it was.
    lines: dict[str, Ranges] = field(default_factory=dict)
    # The PR's files changed with no lines to point at: a binary file, a file mode.
    files: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.lines and not self.files


@dataclass
class _Hunk:
    old_path: str
    new_path: str
    removed: tuple[int, int] | None  # old-side lines, inclusive
    added: tuple[int, int] | None  # new-side lines, inclusive
    at: int  # the new-side line the hunk sits at, for one that adds nothing


def _hunks(patch: str) -> list[_Hunk]:
    found: list[_Hunk] = []
    old_path = new_path = ""
    for line in patch.splitlines():
        if match := _FILE.match(line):
            old_path, new_path = match.group(1), match.group(2)
        elif match := _HUNK.match(line):
            old_start, old_count = int(match.group(1)), int(match.group(2) or 1)
            new_start, new_count = int(match.group(3)), int(match.group(4) or 1)
            found.append(
                _Hunk(
                    old_path,
                    new_path,
                    (old_start, old_start + old_count - 1) if old_count else None,
                    (new_start, new_start + new_count - 1) if new_count else None,
                    max(new_start, 1),
                )
            )
    return found


def _added(patch: str) -> dict[str, Ranges]:
    lines: dict[str, Ranges] = {}
    for hunk in _hunks(patch):
        if hunk.added is not None:
            lines.setdefault(hunk.new_path, []).append(hunk.added)
    return lines


def _overlap(span: tuple[int, int], ranges: Ranges) -> Ranges:
    return [
        (max(span[0], start), min(span[1], end))
        for start, end in ranges
        if start <= span[1] and span[0] <= end
    ]


def _merged(ranges: Ranges) -> Ranges:
    out: Ranges = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def increment(
    git_dir: Path, previous_base: str, previous_head: str, base: str, head: str
) -> Increment | None:
    """The lines to review at `head` since `previous_head` was reviewed; None if unknown.

    `previous_base` and `base` are the default branch's commits each round's PR diff was
    taken against (`base...head`, from where the PR branched off).
    """
    for sha in (previous_base, previous_head, base, head):
        check_commit_sha(sha)

    def _names(span: str) -> list[str]:
        # NUL-separated, so a path is never quoted; a rename gives its new name last.
        args = ("diff", "--name-only", "-z", "-M", "--no-ext-diff", span)
        output = run_git(git_dir, *args, timeout=GIT_TIMEOUT_SECONDS).decode("utf-8", "replace")
        return [p for p in output.split("\0") if p]

    def diff(span: str) -> str:
        # Paths as written: git quotes non-ASCII ones in octal unless told not to.
        args = (
            "-c",
            "core.quotePath=false",
            "diff",
            "-U0",
            "-M",
            "--no-color",
            "--no-ext-diff",
            span,
        )
        return run_git(git_dir, *args, timeout=GIT_TIMEOUT_SECONDS).decode("utf-8", "replace")

    try:
        # A previous head force-pushed away and never fetched cannot be compared.
        for sha in (previous_base, previous_head):
            run_git(git_dir, "cat-file", "-e", f"{sha}^{{commit}}", timeout=GIT_TIMEOUT_SECONDS)
        now = _added(diff(f"{base}...{head}"))
        before = _added(diff(f"{previous_base}...{previous_head}"))
        between = _hunks(diff(f"{previous_head}..{head}"))
        own_files = set(_names(f"{base}...{head}"))
        changed_files = _names(f"{previous_head}..{head}")
    except GitError:
        return None
    lines: dict[str, Ranges] = {}
    for hunk in between:
        own = now.get(hunk.new_path, [])
        kept = _overlap(hunk.added, own) if hunk.added is not None else []
        if kept:
            lines.setdefault(hunk.new_path, []).extend(kept)
        elif hunk.removed is not None and _overlap(hunk.removed, before.get(hunk.old_path, [])):
            # The author took back what the PR had added: review where it was.
            lines.setdefault(hunk.new_path, []).append((hunk.at, hunk.at))
    # A file changed with no text hunk at all is the PR's change too, if the PR changes it.
    texts = {hunk.new_path for hunk in between}
    lineless = sorted(p for p in changed_files if p in own_files and p not in texts)
    merged = {path: _merged(r) for path, r in sorted(lines.items())}
    return Increment(previous_head, merged, lineless)
