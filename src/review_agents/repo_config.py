"""The reviewed repository's `.github/review-agents.yml`, read from its default branch."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import ErrorDetails

CONFIG_PATH = ".github/review-agents.yml"
MAX_CONFIG_BYTES = 64 * 1024

DIMENSIONS = ("spec_drift", "edge_cases", "security", "maintainability", "verifiability")
Dimension = Literal["spec_drift", "edge_cases", "security", "maintainability", "verifiability"]
# What the agents write in and what the report's own text is in (decision D-034).
Language = Literal["zh", "en"]

# Files the review never reads. A repository can add to this list but not remove from it.
DEFAULT_SKIP_PATHS = (
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/poetry.lock",
    "**/uv.lock",
    "**/Cargo.lock",
    "**/go.sum",
    "**/Gemfile.lock",
    "**/composer.lock",
    "**/vendor/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/*.pb.go",
    "**/*.generated.*",
    "**/*.min.js",
    "**/*.min.css",
)
# Binary files, and files above this size, are skipped when they are read.
MAX_FILE_BYTES = 200 * 1024

Glob = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    # Unknown keys are errors, so a misspelt option is reported instead of silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Dimensions(_Strict):
    spec_drift: bool = True
    edge_cases: bool = True
    security: bool = True
    maintainability: bool = True
    verifiability: bool = True


# The longest a review may run and the most tools one dimension agent may call: safety
# nets, not design limits (decisions D-016, D-028); a repository can only lower them.
MAX_REVIEW_MINUTES = 30
MAX_REVIEW_ROUNDS = 10
MAX_TOOL_CALLS_PER_AGENT = 50


class Limits(_Strict):
    """Each limit defaults to the service ceiling and can only be lowered."""

    # The ceiling is the service setting `tokens_per_review` (decision D-009); a value
    # above it is lowered to it, since the repository cannot see what it is.
    tokens_per_review: int | None = Field(default=None, ge=1)
    tool_calls_per_agent: int = Field(
        default=MAX_TOOL_CALLS_PER_AGENT, ge=1, le=MAX_TOOL_CALLS_PER_AGENT
    )
    review_minutes: int = Field(default=MAX_REVIEW_MINUTES, ge=1, le=MAX_REVIEW_MINUTES)
    # Pushes reviewed per PR (D-039); later pushes are not, unless `/review` asks.
    full_review_rounds: int = Field(default=MAX_REVIEW_ROUNDS, ge=1, le=MAX_REVIEW_ROUNDS)
    replies_per_thread: int = Field(default=10, ge=1, le=10)
    nits_per_report: int = Field(default=5, ge=0, le=5)

    def token_budget(self, ceiling: int) -> int:
        """Tokens one review may use: the service ceiling, unless this repository lowers it."""
        return min(self.tokens_per_review or ceiling, ceiling)


class CustomRule(_Strict):
    name: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    branches: list[Glob] = Field(default_factory=list)
    paths: list[Glob] = Field(default_factory=list)


def _inside_repository(path: str) -> str:
    parts = PurePosixPath(path.strip()).parts
    if not parts or path.startswith("/") or "\\" in path or ".." in parts:
        raise ValueError("must be a relative path inside the repository")
    return path


class RepoConfig(_Strict):
    language: Language = "zh"
    mode: Literal["quick", "standard", "deep"] = "standard"
    dimensions: Dimensions = Field(default_factory=Dimensions)
    specs: list[Glob] = Field(default_factory=list)
    skip_paths: list[Glob] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    custom_rules: list[CustomRule] = Field(default_factory=list)
    skill_additions: dict[Dimension, str] = Field(default_factory=dict)

    @field_validator("dimensions")
    @classmethod
    def _at_least_one_dimension(cls, value: Dimensions) -> Dimensions:
        if not any(value.model_dump().values()):
            raise ValueError("enable at least one dimension")
        return value

    @field_validator("skip_paths")
    @classmethod
    def _skip_paths_only_add(cls, value: list[str]) -> list[str]:
        negated = [pattern for pattern in value if pattern.startswith("!")]
        if negated:
            raise ValueError(
                f"patterns can only add to the default skip list, not remove from it: {negated}"
            )
        return value

    @field_validator("skill_additions")
    @classmethod
    def _skill_additions_stay_in_the_repository(cls, value: dict[str, str]) -> dict[str, str]:
        for dimension, path in value.items():
            try:
                _inside_repository(path)
            except ValueError as exc:
                raise ValueError(f"{dimension}: {exc}") from None
        return value

    @property
    def enabled_dimensions(self) -> list[str]:
        return [name for name, enabled in self.dimensions.model_dump().items() if enabled]

    @property
    def effective_skip_paths(self) -> list[str]:
        return [*DEFAULT_SKIP_PATHS, *self.skip_paths]


@dataclass(frozen=True)
class LoadedConfig:
    """The config a review runs with. `errors` is non-empty when the file was rejected."""

    config: RepoConfig = field(default_factory=RepoConfig)
    errors: list[str] = field(default_factory=list)
    # Options that no longer do anything, found in the file and ignored (RETIRED_OPTIONS).
    retired: list[str] = field(default_factory=list)


def _rejected(*errors: str) -> LoadedConfig:
    return LoadedConfig(errors=[*errors, f"{CONFIG_PATH}: using the default configuration instead"])


# Options a config may still have from before; ignored, so the rest of it still counts.
# `block_on_critical`: whether a red result blocks merging is set in GitHub (D-037).
RETIRED_OPTIONS = ("block_on_critical",)


def parse_repo_config(text: str | None) -> LoadedConfig:
    """Validate the file's text; on any error, fall back to the defaults as a whole.

    Falling back as a whole, not field by field, keeps the review from running on a
    half-understood config. The defaults review more, never less.
    """
    if text is None:
        return LoadedConfig()
    if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
        return _rejected(f"{CONFIG_PATH} is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return _rejected(f"{CONFIG_PATH} is not valid YAML: {exc}")
    if data is None:
        return LoadedConfig()
    if not isinstance(data, dict):
        return _rejected(f"{CONFIG_PATH} must be a mapping of options")
    retired = [name for name in RETIRED_OPTIONS if name in data]
    data = {key: value for key, value in data.items() if key not in RETIRED_OPTIONS}
    try:
        return LoadedConfig(config=RepoConfig.model_validate(data), retired=retired)
    except ValidationError as exc:
        return _rejected(*(_describe(error) for error in exc.errors()))


def _describe(error: ErrorDetails) -> str:
    location = ".".join(str(part) for part in error["loc"]) or "(file)"
    message = str(error["msg"]).removeprefix("Value error, ")
    return f"{CONFIG_PATH}: {location}: {message}"


def check_branch_name(name: str) -> str:
    """`name` if git accepts it as a branch, or ValueError."""
    valid = subprocess.run(  # noqa: S603
        ["git", "check-ref-format", f"refs/heads/{name}"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if valid.returncode != 0:
        raise ValueError(f"not a branch name: {name!r}")
    return name


def read_repo_config(repo: Path, default_branch: str) -> LoadedConfig:
    """Read the config from the default branch, never from the pull request.

    A pull request that changes the config does not change how it is reviewed itself.
    """
    # A full branch ref: a tag, a commit or `HEAD` could point at the pull request's own code.
    ref = f"refs/heads/{check_branch_name(default_branch)}"
    result = subprocess.run(  # noqa: S603 - fixed git binary, arguments are not shell-parsed
        ["git", "-C", str(repo), "cat-file", "blob", f"{ref}:{CONFIG_PATH}"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        if _commit_exists(repo, ref):
            return LoadedConfig()
        raise LookupError(f"branch {default_branch!r} not found in {repo}")
    try:
        return parse_repo_config(result.stdout.decode("utf-8"))
    except UnicodeDecodeError:
        return _rejected(f"{CONFIG_PATH} is not UTF-8 text")


def _commit_exists(repo: Path, ref: str) -> bool:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    return result.returncode == 0
