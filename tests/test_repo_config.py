import subprocess
from pathlib import Path

import pytest

from review_agents.repo_config import (
    CONFIG_PATH,
    DEFAULT_SKIP_PATHS,
    MAX_CONFIG_BYTES,
    RepoConfig,
    parse_repo_config,
    read_repo_config,
)


def assert_rejected(text: str, *expected: str) -> list[str]:
    loaded = parse_repo_config(text)
    assert loaded.config == RepoConfig()
    joined = "\n".join(loaded.errors)
    for fragment in expected:
        assert fragment in joined
    assert "using the default configuration instead" in joined
    return loaded.errors


def test_a_missing_or_empty_file_uses_the_defaults_without_errors() -> None:
    for text in (None, "", "# nothing set\n"):
        loaded = parse_repo_config(text)
        assert loaded.config == RepoConfig()
        assert loaded.errors == []


def test_defaults_enable_the_five_dimensions_in_standard_mode() -> None:
    config = RepoConfig()
    assert config.enabled_dimensions == [
        "spec_drift",
        "edge_cases",
        "security",
        "maintainability",
        "verifiability",
    ]
    assert (config.mode, config.language) == ("standard", "zh")


def test_a_full_valid_config_is_read() -> None:
    loaded = parse_repo_config(
        """
language: en
mode: deep
dimensions: {maintainability: false}
specs: [docs/specs/**]
skip_paths: [fixtures/**]
block_on_critical: true
limits: {tokens_per_review: 200000}
custom_rules:
  - name: money-as-integer-cents
    branches: [main]
    paths: [payments/**]
    rule: Amounts are integer cents.
skill_additions:
  security: .github/review-agents/security.md
"""
    )
    assert loaded.errors == []
    config = loaded.config
    assert config.enabled_dimensions == ["spec_drift", "edge_cases", "security", "verifiability"]
    assert config.limits.tokens_per_review == 200_000
    assert config.limits.tool_calls_per_agent == 50
    assert config.custom_rules[0].paths == ["payments/**"]
    assert config.skill_additions == {"security": ".github/review-agents/security.md"}


def test_at_least_one_dimension_must_stay_enabled() -> None:
    off = "\n".join(
        f"  {name}: false"
        for name in ("spec_drift", "edge_cases", "security", "maintainability", "verifiability")
    )
    assert_rejected(f"dimensions:\n{off}\n", "dimensions", "enable at least one dimension")


@pytest.mark.parametrize(
    ("limit", "value"),
    [
        ("tool_calls_per_agent", 51),
        ("review_minutes", 31),
        ("full_review_rounds", 11),
        ("replies_per_thread", 11),
        ("nits_per_report", 6),
        ("tokens_per_review", 0),
    ],
)
def test_limits_can_only_be_lowered(limit: str, value: int) -> None:
    assert_rejected(f"limits: {{{limit}: {value}}}\n", f"limits.{limit}")


def test_the_token_budget_is_the_service_ceiling_unless_the_repository_lowers_it() -> None:
    default = parse_repo_config("").config.limits
    lowered = parse_repo_config("limits: {tokens_per_review: 200000}\n").config.limits
    raised = parse_repo_config("limits: {tokens_per_review: 9000000}\n")

    assert default.token_budget(ceiling=1_500_000) == 1_500_000
    assert lowered.token_budget(ceiling=1_500_000) == 200_000
    assert raised.errors == []
    assert raised.config.limits.token_budget(ceiling=1_500_000) == 1_500_000


def test_a_review_runs_30_minutes_and_an_agent_50_tool_calls_by_default() -> None:
    limits = parse_repo_config("").config.limits  # decision D-028

    assert (limits.review_minutes, limits.tool_calls_per_agent) == (30, 50)


def test_limits_can_be_lowered() -> None:
    loaded = parse_repo_config("limits: {review_minutes: 5, nits_per_report: 0}\n")
    assert loaded.errors == []
    assert loaded.config.limits.review_minutes == 5


def test_skip_paths_add_to_the_defaults_and_cannot_remove_them() -> None:
    config = parse_repo_config("skip_paths: [fixtures/**]\n").config
    assert config.effective_skip_paths == [*DEFAULT_SKIP_PATHS, "fixtures/**"]

    assert_rejected("skip_paths: ['!**/vendor/**']\n", "skip_paths", "only add")


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../other-repo/rules.md", "docs/../../x.md", "docs\\rules.md", "", "  ", "."],
)
def test_skill_additions_must_stay_inside_the_repository(path: str) -> None:
    assert_rejected(f"skill_additions: {{security: '{path}'}}\n", "skill_additions", "security")


def test_skill_additions_only_name_known_dimensions() -> None:
    assert_rejected("skill_additions: {performance: x.md}\n", "skill_additions.performance")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("mode: fast\n", "mode"),
        ("language: fr\n", "language"),
        ("dimensons: {security: false}\n", "dimensons"),
        ("limits: {tokens: 10}\n", "limits.tokens"),
        ("custom_rules: [{name: x}]\n", "custom_rules.0.rule"),
        ("- a list\n", "must be a mapping"),
        ("mode: [unclosed\n", "not valid YAML"),
    ],
)
def test_invalid_configs_name_the_problem_and_fall_back_to_the_defaults(
    text: str, fragment: str
) -> None:
    errors = assert_rejected(text, fragment)
    assert all(error.startswith(CONFIG_PATH) for error in errors)


def test_one_bad_field_rejects_the_whole_file() -> None:
    # Keeping the valid half would run the review on a config nobody wrote.
    loaded = parse_repo_config("dimensions: {security: false}\nmode: fast\n")
    assert loaded.config.dimensions.security is True


def test_an_oversized_file_is_rejected_before_parsing() -> None:
    assert_rejected("#" * (MAX_CONFIG_BYTES + 1), "larger than")


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603, S607


def commit_config(repo: Path, text: str, message: str) -> None:
    path = repo / CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    git(repo, "add", CONFIG_PATH)
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    return tmp_path


def test_a_pull_request_changing_the_config_does_not_change_its_own_review(repo: Path) -> None:
    commit_config(repo, "dimensions: {maintainability: false}\n", "config on main")
    git(repo, "checkout", "-q", "-b", "pr")
    commit_config(repo, "dimensions: {security: false}\n", "pr turns off security")

    loaded = read_repo_config(repo, "main")

    assert loaded.config.dimensions.security is True
    assert loaded.config.dimensions.maintainability is False


def test_a_default_branch_without_the_file_uses_the_defaults(repo: Path) -> None:
    (repo / "README.md").write_text("x", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "init")

    assert read_repo_config(repo, "main").config == RepoConfig()


def test_a_config_that_is_not_utf8_is_rejected(repo: Path) -> None:
    path = repo / CONFIG_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b"mode: \xff\n")
    git(repo, "add", CONFIG_PATH)
    git(repo, "commit", "-q", "-m", "binary config")

    assert "not UTF-8" in "\n".join(read_repo_config(repo, "main").errors)


def test_an_unknown_default_branch_is_an_error_not_the_defaults(repo: Path) -> None:
    commit_config(repo, "mode: quick\n", "init")

    with pytest.raises(LookupError):
        read_repo_config(repo, "trunk")


@pytest.mark.parametrize(
    ("revision", "error"),
    [("HEAD", LookupError), ("pr~0", ValueError), ("pr^", ValueError), ("-p", LookupError)],
)
def test_only_a_branch_is_read_never_another_revision(
    repo: Path, revision: str, error: type[Exception]
) -> None:
    commit_config(repo, "mode: quick\n", "init")
    git(repo, "checkout", "-q", "-b", "pr")
    commit_config(repo, "dimensions: {security: false}\n", "pr turns off security")

    with pytest.raises(error):
        read_repo_config(repo, revision)


def test_a_tag_named_like_the_branch_is_not_read(repo: Path) -> None:
    commit_config(repo, "mode: quick\n", "init")
    git(repo, "checkout", "-q", "-b", "pr")
    commit_config(repo, "dimensions: {security: false}\n", "pr turns off security")
    git(repo, "tag", "main")

    assert read_repo_config(repo, "main").config.dimensions.security is True


def test_a_pr_is_reviewed_ten_rounds_at_most_by_default() -> None:
    # D-039: a PR is often pushed many times; past the limit pushes are not reviewed.
    assert RepoConfig().limits.full_review_rounds == 10


@pytest.mark.parametrize("value", ["true", "false", "maybe"])
def test_a_retired_option_is_ignored_not_an_error(value: str) -> None:
    # D-037: whether a red result blocks merging is set in GitHub's branch protection.
    # A config that still has the option keeps the rest of what it says.
    loaded = parse_repo_config(f"mode: deep\nblock_on_critical: {value}\n")

    assert loaded.errors == [] and loaded.config.mode == "deep"
    assert loaded.retired == ["block_on_critical"]
