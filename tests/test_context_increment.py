"""What a round reviews: the PR's own changes since the last review (issue #12, D-026)."""

from pathlib import Path

import pytest
from gitrepo import Upstream, git

from review_agents.context.increment import Increment, increment
from review_agents.mirror import MirrorStore

LINES = "".join(f"line {n}\n" for n in range(1, 21))


class Repo:
    def __init__(self, tmp_path: Path) -> None:
        self.up = Upstream(tmp_path, "octo/app")
        self.base = self.up.commit({"a.py": LINES, "b.py": LINES}, "init")
        git(self.up.work, "checkout", "-q", "-b", "pr")
        self.store = MirrorStore(tmp_path / "repos", remote_base=str(self.up.base))

    def edit(self, path: str, change: dict[int, str | None], branch: str = "pr") -> str:
        """Replace (or with None, delete) numbered lines of a file, and commit."""
        text = (self.up.work / path).read_text().splitlines(keepends=True)
        for number in sorted(change, reverse=True):
            new = change[number]
            if new is None:
                del text[number - 1]
            else:
                text[number - 1] = new
        return self.up.commit({path: "".join(text)}, f"edit {path}", branch=branch)

    def main(self, files: dict[str, str | bytes]) -> str:
        git(self.up.work, "checkout", "-q", "main")
        sha = self.up.commit(files, "main moves on")
        git(self.up.work, "checkout", "-q", "pr")
        return sha

    def sync(self) -> Path:
        refs = ["refs/heads/main", "refs/heads/pr"]
        return self.store.sync("octo/app", refs, [], token=None)

    def head(self, ref: str = "pr") -> str:
        return git(self.up.work, "rev-parse", ref)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path)


def test_a_new_commit_s_lines_are_the_increment(repo: Repo) -> None:
    first = repo.edit("a.py", {3: "changed 3\n"})
    second = repo.edit("a.py", {10: "changed 10\n", 11: "changed 11\n"})

    found = increment(repo.sync(), repo.base, first, repo.base, second)

    assert found == Increment(first, {"a.py": [(10, 11)]})
    assert not found.empty


def test_a_line_changed_again_is_in_the_increment(repo: Repo) -> None:
    first = repo.edit("a.py", {3: "changed 3\n"})
    second = repo.edit("a.py", {3: "changed 3 again\n"})

    found = increment(repo.sync(), repo.base, first, repo.base, second)

    assert found is not None and found.lines == {"a.py": [(3, 3)]}


def test_merging_the_default_branch_in_brings_nothing_to_review(repo: Repo) -> None:
    first = repo.edit("a.py", {3: "changed 3\n"})
    moved = repo.main({"b.py": LINES.replace("line 7\n", "theirs 7\n"), "c.py": "new\n"})
    git(repo.up.work, "merge", "-q", "--no-edit", "main")
    git(repo.up.work, "push", "-q", "origin", "HEAD:refs/heads/pr")

    found = increment(repo.sync(), repo.base, first, moved, repo.head())

    assert found is not None and found.empty


def test_a_rebase_with_no_change_of_its_own_brings_nothing_to_review(repo: Repo) -> None:
    first = repo.edit("a.py", {3: "changed 3\n"})
    repo.sync()  # the previous round fetched the head it reviewed
    moved = repo.main({"b.py": LINES.replace("line 7\n", "theirs 7\n")})
    git(repo.up.work, "rebase", "-q", "main")
    git(repo.up.work, "push", "-q", "-f", "origin", "HEAD:refs/heads/pr")

    found = increment(repo.sync(), repo.base, first, moved, repo.head())

    assert found is not None and found.empty


def test_a_rebase_with_a_change_of_its_own_reviews_only_that(repo: Repo) -> None:
    first = repo.edit("a.py", {3: "changed 3\n"})
    repo.sync()
    moved = repo.main({"b.py": LINES.replace("line 7\n", "theirs 7\n")})
    git(repo.up.work, "rebase", "-q", "main")
    git(repo.up.work, "push", "-q", "-f", "origin", "HEAD:refs/heads/pr")
    repo.edit("a.py", {15: "changed 15\n"})

    found = increment(repo.sync(), repo.base, first, moved, repo.head())

    assert found is not None and found.lines == {"a.py": [(15, 15)]}


def test_taking_back_an_earlier_change_is_a_change_to_review(repo: Repo) -> None:
    # The fix may be removing what the PR added: nothing is added, but it was the
    # author's change, so the place is reviewed.
    first = repo.edit("a.py", {5: "line 5\nadded by the pr\n"})
    second = repo.edit("a.py", {6: None})

    found = increment(repo.sync(), repo.base, first, repo.base, second)

    assert found is not None and found.lines == {"a.py": [(5, 5)]}


def test_without_the_previous_head_the_round_reviews_everything(repo: Repo) -> None:
    head = repo.edit("a.py", {3: "changed 3\n"})
    gone = "f" * 40

    assert increment(repo.sync(), repo.base, gone, repo.base, head) is None


def test_a_file_renamed_since_the_last_review_keeps_only_its_new_lines(repo: Repo) -> None:
    first = repo.edit("a.py", {3: "changed 3\n"})
    git(repo.up.work, "mv", "a.py", "renamed.py")
    repo.up.commit({}, "rename", branch="pr")
    second = repo.edit("renamed.py", {12: "changed 12\n"})

    found = increment(repo.sync(), repo.base, first, repo.base, second)

    assert found is not None and found.lines == {"renamed.py": [(12, 12)]}


def test_a_path_in_chinese_is_named_as_written(repo: Repo) -> None:
    # git quotes such paths in octal by default; the PR layer names them as written.
    repo.up.commit({"文档/说明.py": LINES}, "add", branch="pr")
    first = repo.edit("文档/说明.py", {3: "changed 3\n"})
    second = repo.edit("文档/说明.py", {9: "changed 9\n"})

    found = increment(repo.sync(), repo.base, first, repo.base, second)

    assert found is not None and found.lines == {"文档/说明.py": [(9, 9)]}


@pytest.mark.parametrize("kind", ["binary", "mode"])
def test_a_change_with_no_lines_is_still_a_change_to_review(repo: Repo, kind: str) -> None:
    # A binary file or a file mode has no lines to point at, but it is the PR's change.
    repo.up.commit({"logo.png": b"\x89PNG\x00one"}, "add", branch="pr")
    first = repo.edit("a.py", {3: "changed 3\n"})
    if kind == "binary":
        second = repo.up.commit({"logo.png": b"\x89PNG\x00two"}, "redraw", branch="pr")
    else:
        git(repo.up.work, "update-index", "--chmod=+x", "a.py")
        git(repo.up.work, "commit", "-q", "-m", "make it executable")
        git(repo.up.work, "push", "-q", "origin", "HEAD:refs/heads/pr")
        second = repo.head()
    path = "logo.png" if kind == "binary" else "a.py"

    found = increment(repo.sync(), repo.base, first, repo.base, second)

    assert found is not None and not found.empty
    assert found.lines == {} and found.files == [path]
