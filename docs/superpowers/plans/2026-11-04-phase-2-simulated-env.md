# Phase 2 — Simulated Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four JSON/dict-backed simulated apps (`repo`, `chat`, `pager`, `cal`), the allowlist command parser, checkpoint isolation (fresh deep copy), and the deterministic L0–L3 action-level classifier — with offline unit tests and a scripted tool-trace integration test.

**Architecture:** Each app is a module exporting `initial_state()` and a `TOOLS` tuple of `ToolSpec(app, name, kind, handler, required)`. A shared `base` module holds the spec/error types; `registry` aggregates the four apps; `parser` turns raw `{app, function, parameter}` dicts into validated `Command`s; `workspace` owns per-checkpoint state with deepcopy isolation and executes commands; `classifier` maps a final action to `L0`–`L3` via registry kind + an explicit write-tool table.

**Tech Stack:** Python 3.11+, stdlib only (`copy`, `datetime`, `dataclasses`, `typing`), pytest, ruff.

**Spec:** [`SPEC.md`](../../../SPEC.md) — §3.1 (apps + rules), §3.4 (action levels), §3.5 (event/command shape), §11 Phase 2 row.

## Global Constraints

- Python 3.11+; ruff `select = ["E", "F", "I", "UP", "B"]`, `line-length = 100` (pyproject.toml).
- Tests run fully offline — no network, no real git, no real Slack (SPEC §3.1, §9).
- State is plain JSON/dict; each checkpoint gets a **fresh deep copy** (SPEC §3.1).
- Tools are validated against an allowlist; unknown tool raises a `ToolError` subclass (bounded retry belongs to the Phase 4 agent loop; Phase 2 only raises) (SPEC §3.1).
- Classifier: deterministic function of final tool + args, unit-tested, not LLM-judged (SPEC §3.4).
- Action levels (SPEC §3.4): L0 observe/no intervention (includes read-only and `None`), L1 notify/inform, L2 prepare/draft, L3 execute.
- The allowlist must match SPEC §3.1 exactly — 32 tools:

| App | Reads (inspect) | Writes (final) |
|-----|-----------------|----------------|
| `repo` | `list_prs`, `get_pr`, `get_checks`, `list_issues`, `get_issue` | `post_comment`, `request_review`, `retrigger_ci`, `open_draft_pr`, `merge_pr`, `revert_commit` |
| `chat` | `list_channels`, `get_thread`, `search_messages` | `send_message`, `post_standup_update`, `pin_message` |
| `pager` | `list_alerts`, `get_alert`, `get_runbook`, `get_deploy_window` | `ack_alert`, `page_oncall`, `propose_rollback`, `execute_rollback` |
| `cal` | `list_events`, `get_event`, `get_free_slots` | `create_event`, `decline_event`, `propose_reschedule`, `block_focus_time` |

- Write-tool → action level (deterministic; spec gives examples, this table completes it):

| Level | Write tools |
|-------|-------------|
| L1 | `post_comment`, `request_review`, `retrigger_ci`, `send_message`, `post_standup_update`, `pin_message`, `ack_alert`, `page_oncall` |
| L2 | `open_draft_pr`, `propose_rollback`, `propose_reschedule`, `block_focus_time` |
| L3 | `merge_pr`, `revert_commit`, `execute_rollback`, `create_event`, `decline_event` |

- No new runtime dependencies (httpx/pyyaml/dotenv already present; Phase 2 needs only stdlib).
- Commit message style: `Phase 2: <component>` (matches Phase 0/1 history).
- Secrets never committed (n/a for this phase; `.env` stays gitignored).

---

### Task 1: `env/base.py` — tool spec, errors, id helper

**Files:**
- Create: `src/proactive_bench/env/base.py`
- Test: `tests/test_env_base.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces (used by every later task):
  - `ToolKind = Literal["read", "write"]`
  - `Handler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]`
  - `class ToolError(Exception)` with subclasses `InvalidCommandError`, `UnknownAppError`, `UnknownToolError`, `MissingParameterError`, `ToolKindError`, `ToolExecutionError`
  - `@dataclass(frozen=True) class ToolSpec(app: str, name: str, kind: ToolKind, handler: Handler, required: tuple[str, ...] = ())` with method `check_params(params: Mapping[str, Any]) -> None`
  - `next_id(items: list[dict[str, Any]], prefix: str) -> str`
  - `as_int(value: Any, name: str) -> int`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for env base types: ToolSpec, errors, helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from proactive_bench.env.base import (
    InvalidCommandError,
    MissingParameterError,
    ToolError,
    ToolExecutionError,
    ToolKindError,
    ToolSpec,
    UnknownAppError,
    UnknownToolError,
    as_int,
    next_id,
)


def _noop_handler(state: dict, params: dict) -> dict:
    return {"ok": True}


def make_spec(**kwargs: object) -> ToolSpec:
    defaults: dict = {
        "app": "repo",
        "name": "get_pr",
        "kind": "read",
        "handler": _noop_handler,
        "required": ("number",),
    }
    defaults.update(kwargs)
    return ToolSpec(**defaults)


def test_errors_are_tool_error_subclasses() -> None:
    for cls in (
        InvalidCommandError,
        UnknownAppError,
        UnknownToolError,
        MissingParameterError,
        ToolKindError,
        ToolExecutionError,
    ):
        assert issubclass(cls, ToolError)
        assert issubclass(cls, Exception)


def test_check_params_ok_when_present() -> None:
    make_spec().check_params({"number": 1})


def test_check_params_lists_missing_keys() -> None:
    with pytest.raises(MissingParameterError) as exc:
        make_spec(required=("number", "body")).check_params({"number": 1})
    assert "body" in str(exc.value)
    assert "number" not in str(exc.value).split(":")[-1]


def test_tool_spec_is_immutable() -> None:
    spec = make_spec()
    with pytest.raises(FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_next_id_increments_numeric_suffix() -> None:
    items = [{"id": "m1"}, {"id": "m2"}, {"id": "other"}]
    assert next_id(items, "m") == "m3"


def test_next_id_starts_at_one_when_empty() -> None:
    assert next_id([], "e") == "e1"


def test_next_id_skips_non_matching_prefix() -> None:
    items = [{"id": "e2"}, {"id": "f9"}]
    assert next_id(items, "e") == "e3"


def test_next_id_with_empty_prefix() -> None:
    items = [{"id": "1"}, {"id": "4"}, {"id": "x"}]
    assert next_id(items, "") == "5"


def test_as_int_accepts_int_and_digit_string() -> None:
    assert as_int(7, "number") == 7
    assert as_int("7", "number") == 7
    assert as_int("-7", "number") == -7
    assert as_int(" 7 ", "number") == 7


def test_as_int_rejects_garbage() -> None:
    for bad in (None, 1.5, True, "abc", [1], "--7", "²"):
        with pytest.raises(InvalidCommandError):
            as_int(bad, "number")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.base'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Shared tool specification, error types, and helpers for simulated apps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

ToolKind = Literal["read", "write"]

Handler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class ToolError(Exception):
    """Base class for simulated-app failures surfaced to the agent."""


class InvalidCommandError(ToolError):
    """The command payload is structurally invalid or has bad value types."""


class UnknownAppError(ToolError):
    """The requested app id is not part of the benchmark."""


class UnknownToolError(ToolError):
    """The requested function is not in the app's allowlist."""


class MissingParameterError(ToolError):
    """A required parameter was absent from the command."""


class ToolKindError(ToolError):
    """A read tool was used where a write is required, or vice versa."""


class ToolExecutionError(ToolError):
    """The tool ran but could not complete (e.g. entity not found)."""


@dataclass(frozen=True)
class ToolSpec:
    """One allowlisted tool: metadata plus its handler."""

    app: str
    name: str
    kind: ToolKind
    handler: Handler
    required: tuple[str, ...] = ()

    @property
    def full_name(self) -> str:
        return f"{self.app}.{self.name}"

    def check_params(self, params: Mapping[str, Any]) -> None:
        missing = [key for key in self.required if key not in params]
        if missing:
            raise MissingParameterError(
                f"{self.full_name} missing required parameter(s): {', '.join(missing)}"
            )


def next_id(items: list[dict[str, Any]], prefix: str) -> str:
    """Next ``<prefix><n>`` id, one past the max numeric suffix already present."""
    highest = 0
    for item in items:
        raw = str(item.get("id", ""))
        if raw.startswith(prefix) and raw[len(prefix) :].isdigit():
            highest = max(highest, int(raw[len(prefix) :]))
    return f"{prefix}{highest + 1}"


def as_int(value: Any, name: str) -> int:
    """Coerce a JSON integer or digit string to ``int``; else raise."""
    if isinstance(value, bool):
        raise InvalidCommandError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.removeprefix("-").isdecimal():
            return int(s)
    raise InvalidCommandError(f"{name} must be an integer, got {value!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_base.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/base.py tests/test_env_base.py
git commit -m "Phase 2: env base types and errors"
```

---

### Task 2: `env/repo.py` — GitHub-like app

**Files:**
- Create: `src/proactive_bench/env/repo.py`
- Test: `tests/test_env_repo.py`

**Interfaces:**
- Consumes: `ToolSpec`, `ToolExecutionError`, `InvalidCommandError`, `as_int` from Task 1.
- Produces:
  - `initial_state() -> dict[str, Any]` returning `{"prs": [], "issues": [], "commits": []}`
  - `TOOLS: tuple[ToolSpec, ...]` — 11 tools, names/kinds/required exactly:
    - reads: `list_prs()`, `get_pr(number)`, `get_checks(number)`, `list_issues()`, `get_issue(number)`
    - writes: `post_comment(number, body)`, `request_review(number, reviewer)`, `retrigger_ci(number)`, `open_draft_pr(title, branch)`, `merge_pr(number)` (optional bool `force`), `revert_commit(sha)`
  - PR dict shape: `{"number", "title", "branch", "base", "draft", "status", "checks", "reviewers", "comments", "author"}`
  - Check dict shape: `{"name", "status"}` where status ∈ `pass|fail|pending`
  - Issue dict shape: `{"number", "title", "state", "labels", "comments"}`
  - Commit dict shape: `{"sha", "message", "reverted"}`
  - Merge rule: any check whose status is not `"pass"` blocks the merge unless `force=True`; a draft PR never merges; a PR with zero checks merges.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the simulated repo app (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import repo
from proactive_bench.env.base import InvalidCommandError, ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in repo.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def base_pr(**overrides: object) -> dict:
    pr = {
        "number": 1,
        "title": "fix thing",
        "branch": "fix-1",
        "base": "main",
        "draft": False,
        "status": "open",
        "checks": [{"name": "ci", "status": "pass"}],
        "reviewers": [],
        "comments": [],
        "author": "dev",
    }
    pr.update(overrides)
    return pr


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in repo.TOOLS if s.kind == "read"}
    writes = {s.name for s in repo.TOOLS if s.kind == "write"}
    assert reads == {"list_prs", "get_pr", "get_checks", "list_issues", "get_issue"}
    assert writes == {
        "post_comment",
        "request_review",
        "retrigger_ci",
        "open_draft_pr",
        "merge_pr",
        "revert_commit",
    }


def test_initial_state_is_fresh_each_call() -> None:
    a, b = repo.initial_state(), repo.initial_state()
    assert a == {"prs": [], "issues": [], "commits": []}
    a["prs"].append(1)
    assert b["prs"] == []


def test_list_and_get_pr() -> None:
    state = {"prs": [base_pr()]}
    assert get("list_prs").handler(state, {}) == {"prs": state["prs"]}
    result = get("get_pr").handler(state, {"number": 1})
    assert result["pr"]["title"] == "fix thing"
    assert get("get_pr").handler(state, {"number": "1"})["pr"]["number"] == 1
    with pytest.raises(ToolExecutionError):
        get("get_pr").handler(state, {"number": 99})


def test_get_checks_and_list_issues() -> None:
    state = {
        "prs": [base_pr()],
        "issues": [{"number": 5, "title": "bug", "state": "open", "labels": [], "comments": []}],
    }
    assert get("get_checks").handler(state, {"number": 1})["checks"][0]["status"] == "pass"
    assert get("list_issues").handler(state, {})["issues"][0]["number"] == 5
    assert get("get_issue").handler(state, {"number": 5})["issue"]["title"] == "bug"
    with pytest.raises(ToolExecutionError):
        get("get_issue").handler(state, {"number": 6})


def test_post_comment_appends_to_pr_and_issue() -> None:
    state = {
        "prs": [base_pr()],
        "issues": [{"number": 5, "title": "bug", "state": "open", "labels": [], "comments": []}],
    }
    result = get("post_comment").handler(state, {"number": 1, "body": "lgtm"})
    assert result["ok"] is True
    assert state["prs"][0]["comments"] == [{"author": "agent", "body": "lgtm"}]
    get("post_comment").handler(state, {"number": 5, "body": "repro?"})
    assert len(state["issues"][0]["comments"]) == 1
    with pytest.raises(ToolExecutionError):
        get("post_comment").handler(state, {"number": 404, "body": "x"})


def test_request_review_deduplicates() -> None:
    state = {"prs": [base_pr(reviewers=["alice"])]}
    get("request_review").handler(state, {"number": 1, "reviewer": "alice"})
    get("request_review").handler(state, {"number": 1, "reviewer": "bob"})
    assert state["prs"][0]["reviewers"] == ["alice", "bob"]


def test_retrigger_ci_sets_checks_pending() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "fail"}])]}
    get("retrigger_ci").handler(state, {"number": 1})
    assert state["prs"][0]["checks"][0]["status"] == "pending"


def test_open_draft_pr_assigns_next_number() -> None:
    state = {"prs": [base_pr(number=3)]}
    result = get("open_draft_pr").handler(state, {"title": "new", "branch": "feat-x"})
    pr = result["pr"]
    assert pr["number"] == 4
    assert pr["draft"] is True
    assert pr["status"] == "open"
    assert pr["base"] == "main"
    assert pr["checks"] == [] and pr["reviewers"] == [] and pr["comments"] == []
    assert state["prs"][-1]["title"] == "new"


def test_merge_pr_blocked_by_non_passing_checks() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "fail"}])]}
    with pytest.raises(ToolExecutionError, match="failing or pending"):
        get("merge_pr").handler(state, {"number": 1})
    assert state["prs"][0]["status"] == "open"


def test_merge_pr_blocked_by_pending_checks_unless_forced() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "pending"}])]}
    with pytest.raises(ToolExecutionError):
        get("merge_pr").handler(state, {"number": 1})
    result = get("merge_pr").handler(state, {"number": 1, "force": True})
    assert result["ok"] is True
    assert state["prs"][0]["status"] == "merged"


def test_merge_pr_happy_path_and_draft_block() -> None:
    state = {"prs": [base_pr(checks=[]), base_pr(number=2, draft=True, checks=[])]}
    get("merge_pr").handler(state, {"number": 1})
    assert state["prs"][0]["status"] == "merged"
    with pytest.raises(ToolExecutionError, match="draft"):
        get("merge_pr").handler(state, {"number": 2})


def test_merge_pr_rejects_non_bool_force() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "fail"}])]}
    with pytest.raises(InvalidCommandError, match="force"):
        get("merge_pr").handler(state, {"number": 1, "force": "yes"})


def test_revert_commit_marks_and_blocks_double_revert() -> None:
    state = {"commits": [{"sha": "abc123", "message": "wip", "reverted": False}]}
    get("revert_commit").handler(state, {"sha": "abc123"})
    assert state["commits"][0]["reverted"] is True
    with pytest.raises(ToolExecutionError, match="already reverted"):
        get("revert_commit").handler(state, {"sha": "abc123"})
    with pytest.raises(ToolExecutionError):
        get("revert_commit").handler(state, {"sha": "nope"})


def test_required_params_enforced() -> None:
    from proactive_bench.env.base import MissingParameterError

    with pytest.raises(MissingParameterError, match="missing required parameter"):
        get("get_pr").check_params({})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_repo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.repo'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Simulated GitHub-like repo app: PRs, checks, issues, commits."""

from __future__ import annotations

from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    as_int,
)


def initial_state() -> dict[str, Any]:
    return {"prs": [], "issues": [], "commits": []}


def _find_pr(state: dict[str, Any], number: Any) -> dict[str, Any]:
    wanted = as_int(number, "number")
    for pr in state.setdefault("prs", []):
        if pr.get("number") == wanted:
            return pr
    raise ToolExecutionError(f"no PR with number {wanted}")


def _find_issue(state: dict[str, Any], number: Any) -> dict[str, Any]:
    wanted = as_int(number, "number")
    for issue in state.setdefault("issues", []):
        if issue.get("number") == wanted:
            return issue
    raise ToolExecutionError(f"no issue with number {wanted}")


def _find_commit(state: dict[str, Any], sha: Any) -> dict[str, Any]:
    if not isinstance(sha, str) or not sha:
        raise InvalidCommandError(f"sha must be a non-empty string, got {sha!r}")
    for commit in state.setdefault("commits", []):
        if commit.get("sha") == sha:
            return commit
    raise ToolExecutionError(f"no commit with sha {sha!r}")


def _find_commentable(state: dict[str, Any], number: Any) -> dict[str, Any]:
    wanted = as_int(number, "number")
    for entity in [*state.setdefault("prs", []), *state.setdefault("issues", [])]:
        if entity.get("number") == wanted:
            return entity
    raise ToolExecutionError(f"no PR or issue with number {wanted}")


# --- reads -----------------------------------------------------------------


def _list_prs(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"prs": list(state.setdefault("prs", []))}


def _get_pr(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"pr": _find_pr(state, params["number"])}


def _get_checks(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"checks": list(_find_pr(state, params["number"]).get("checks", []))}


def _list_issues(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"issues": list(state.setdefault("issues", []))}


def _get_issue(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"issue": _find_issue(state, params["number"])}


# --- writes ----------------------------------------------------------------


def _post_comment(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    body = params["body"]
    if not isinstance(body, str) or not body:
        raise InvalidCommandError(f"body must be a non-empty string, got {body!r}")
    entity = _find_commentable(state, params["number"])
    comment = {"author": "agent", "body": body}
    entity.setdefault("comments", []).append(comment)
    return {"ok": True, "comment": comment}


def _request_review(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    reviewer = params["reviewer"]
    if not isinstance(reviewer, str) or not reviewer:
        raise InvalidCommandError(f"reviewer must be a non-empty string, got {reviewer!r}")
    pr = _find_pr(state, params["number"])
    reviewers = pr.setdefault("reviewers", [])
    if reviewer not in reviewers:
        reviewers.append(reviewer)
    return {"ok": True, "reviewers": list(reviewers)}


def _retrigger_ci(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    pr = _find_pr(state, params["number"])
    for check in pr.setdefault("checks", []):
        check["status"] = "pending"
    return {"ok": True, "checks": list(pr.get("checks", []))}


def _open_draft_pr(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    title = params["title"]
    branch = params["branch"]
    for key, value in (("title", title), ("branch", branch)):
        if not isinstance(value, str) or not value:
            raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    prs = state.setdefault("prs", [])
    number = max((pr.get("number", 0) for pr in prs), default=0) + 1
    pr = {
        "number": number,
        "title": title,
        "branch": branch,
        "base": "main",
        "draft": True,
        "status": "open",
        "checks": [],
        "reviewers": [],
        "comments": [],
        "author": "agent",
    }
    prs.append(pr)
    return {"ok": True, "pr": pr}


def _merge_pr(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    pr = _find_pr(state, params["number"])
    force = params.get("force", False)
    if not isinstance(force, bool):
        raise InvalidCommandError(f"force must be a boolean, got {force!r}")
    if pr.get("draft"):
        raise ToolExecutionError(f"cannot merge PR #{pr['number']}: it is a draft")
    not_passing = [c for c in pr.get("checks", []) if c.get("status") != "pass"]
    if not_passing and not force:
        names = [c.get("name", "?") for c in not_passing]
        raise ToolExecutionError(
            f"cannot merge PR #{pr['number']}: failing or pending checks {names} "
            "(force=true to override)"
        )
    pr["status"] = "merged"
    pr["draft"] = False
    return {"ok": True, "pr": pr}


def _revert_commit(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    commit = _find_commit(state, params["sha"])
    if commit.get("reverted"):
        raise ToolExecutionError(f"commit {commit['sha']} already reverted")
    commit["reverted"] = True
    return {"ok": True, "commit": commit}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="repo", name="list_prs", kind="read", handler=_list_prs),
    ToolSpec(app="repo", name="get_pr", kind="read", handler=_get_pr, required=("number",)),
    ToolSpec(
        app="repo", name="get_checks", kind="read", handler=_get_checks, required=("number",)
    ),
    ToolSpec(app="repo", name="list_issues", kind="read", handler=_list_issues),
    ToolSpec(app="repo", name="get_issue", kind="read", handler=_get_issue, required=("number",)),
    ToolSpec(
        app="repo",
        name="post_comment",
        kind="write",
        handler=_post_comment,
        required=("number", "body"),
    ),
    ToolSpec(
        app="repo",
        name="request_review",
        kind="write",
        handler=_request_review,
        required=("number", "reviewer"),
    ),
    ToolSpec(
        app="repo", name="retrigger_ci", kind="write", handler=_retrigger_ci, required=("number",)
    ),
    ToolSpec(
        app="repo",
        name="open_draft_pr",
        kind="write",
        handler=_open_draft_pr,
        required=("title", "branch"),
    ),
    ToolSpec(app="repo", name="merge_pr", kind="write", handler=_merge_pr, required=("number",)),
    ToolSpec(
        app="repo", name="revert_commit", kind="write", handler=_revert_commit, required=("sha",)
    ),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_repo.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/repo.py tests/test_env_repo.py
git commit -m "Phase 2: repo app tools with tests"
```

---

### Task 3: `env/chat.py` — team chat app

**Files:**
- Create: `src/proactive_bench/env/chat.py`
- Test: `tests/test_env_chat.py`

**Interfaces:**
- Consumes: `ToolSpec`, `ToolExecutionError`, `InvalidCommandError`, `next_id` from Task 1.
- Produces:
  - `initial_state() -> dict[str, Any]` returning `{"channels": []}`
  - `TOOLS: tuple[ToolSpec, ...]` — 6 tools:
    - reads: `list_channels()`, `get_thread(channel, message_id)`, `search_messages(query)`
    - writes: `send_message(channel, text)`, `post_standup_update(channel, text)`, `pin_message(channel, message_id)`
  - Channel shape: `{"name", "messages": [...]}`; message shape: `{"id", "author", "text", "pinned"}` plus optional `"thread_id"` and optional `"kind"` (e.g. `"standup"`).
  - `list_channels` returns `{"channels": [{"name", "message_count"}]}`.
  - `get_thread(channel, message_id)` returns `{"messages": [...]}` — the root plus every reply sharing its thread root (root's own `thread_id` defaults to its `id`).
  - `search_messages` case-insensitive substring over message text → `{"matches": [{"channel", "message"}]}`.
  - `send_message`/`post_standup_update` require the channel to exist (else `ToolExecutionError`); ids come from `next_id(messages, "m")`; standup messages get `"kind": "standup"`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the simulated chat app (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import chat
from proactive_bench.env.base import ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in chat.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def seed_state() -> dict:
    return {
        "channels": [
            {
                "name": "eng",
                "messages": [
                    {"id": "m1", "author": "alice", "text": "CI is red", "pinned": False},
                    {
                        "id": "m2",
                        "author": "bob",
                        "text": "looking",
                        "pinned": False,
                        "thread_id": "m1",
                    },
                    {
                        "id": "m3",
                        "author": "carol",
                        "text": "same here",
                        "pinned": False,
                        "thread_id": "m1",
                    },
                    {"id": "m5", "author": "dan", "text": "lunch?", "pinned": False},
                ],
            },
            {"name": "random", "messages": []},
        ]
    }


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in chat.TOOLS if s.kind == "read"}
    writes = {s.name for s in chat.TOOLS if s.kind == "write"}
    assert reads == {"list_channels", "get_thread", "search_messages"}
    assert writes == {"send_message", "post_standup_update", "pin_message"}


def test_initial_state_is_fresh_each_call() -> None:
    a, b = chat.initial_state(), chat.initial_state()
    assert a == {"channels": []}
    a["channels"].append("x")
    assert b["channels"] == []


def test_list_channels_reports_counts() -> None:
    result = get("list_channels").handler(seed_state(), {})
    assert result["channels"] == [
        {"name": "eng", "message_count": 4},
        {"name": "random", "message_count": 0},
    ]


def test_get_thread_returns_root_and_replies() -> None:
    state = seed_state()
    result = get("get_thread").handler(state, {"channel": "eng", "message_id": "m1"})
    assert [m["id"] for m in result["messages"]] == ["m1", "m2", "m3"]
    from_reply = get("get_thread").handler(state, {"channel": "eng", "message_id": "m2"})
    assert [m["id"] for m in from_reply["messages"]] == ["m1", "m2", "m3"]
    alone = get("get_thread").handler(state, {"channel": "eng", "message_id": "m5"})
    assert [m["id"] for m in alone["messages"]] == ["m5"]


def test_get_thread_unknown_channel_or_message() -> None:
    state = seed_state()
    with pytest.raises(ToolExecutionError, match="channel"):
        get("get_thread").handler(state, {"channel": "nope", "message_id": "m1"})
    with pytest.raises(ToolExecutionError, match="message"):
        get("get_thread").handler(state, {"channel": "eng", "message_id": "m99"})


def test_search_messages_case_insensitive() -> None:
    result = get("search_messages").handler(seed_state(), {"query": "CI"})
    assert [m["message"]["id"] for m in result["matches"]] == ["m1"]
    assert get("search_messages").handler(seed_state(), {"query": "zzz"})["matches"] == []


def test_send_message_appends_with_next_id() -> None:
    state = seed_state()
    result = get("send_message").handler(state, {"channel": "eng", "text": "on it"})
    assert result["ok"] is True
    assert result["message"]["id"] == "m6"
    assert result["message"]["author"] == "agent"
    assert state["channels"][0]["messages"][-1]["text"] == "on it"
    with pytest.raises(ToolExecutionError, match="channel"):
        get("send_message").handler(state, {"channel": "nope", "text": "x"})


def test_post_standup_update_marks_kind() -> None:
    state = seed_state()
    result = get("post_standup_update").handler(
        state, {"channel": "eng", "text": "yesterday: shipped"}
    )
    assert result["message"]["kind"] == "standup"
    assert result["message"]["author"] == "agent"


def test_pin_message_sets_pinned() -> None:
    state = seed_state()
    get("pin_message").handler(state, {"channel": "eng", "message_id": "m5"})
    assert state["channels"][0]["messages"][3]["pinned"] is True
    with pytest.raises(ToolExecutionError):
        get("pin_message").handler(state, {"channel": "eng", "message_id": "m99"})


def test_send_message_rejects_empty_text() -> None:
    from proactive_bench.env.base import InvalidCommandError

    with pytest.raises(InvalidCommandError):
        get("send_message").handler(seed_state(), {"channel": "eng", "text": ""})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.chat'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Simulated team-chat app: channels, messages, threads, pins."""

from __future__ import annotations

from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    next_id,
)


def initial_state() -> dict[str, Any]:
    return {"channels": []}


def _find_channel(state: dict[str, Any], name: Any) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise InvalidCommandError(f"channel must be a non-empty string, got {name!r}")
    for channel in state.setdefault("channels", []):
        if channel.get("name") == name:
            return channel
    known = ", ".join(str(c.get("name")) for c in state.get("channels", [])) or "none"
    raise ToolExecutionError(f"unknown channel {name!r} (known: {known})")


def _find_message(channel: dict[str, Any], message_id: Any) -> dict[str, Any]:
    if not isinstance(message_id, str) or not message_id:
        raise InvalidCommandError(f"message_id must be a non-empty string, got {message_id!r}")
    for message in channel.setdefault("messages", []):
        if message.get("id") == message_id:
            return message
    raise ToolExecutionError(f"unknown message {message_id!r} in channel {channel.get('name')!r}")


def _require_text(params: dict[str, Any]) -> str:
    text = params["text"]
    if not isinstance(text, str) or not text:
        raise InvalidCommandError(f"text must be a non-empty string, got {text!r}")
    return text


# --- reads -----------------------------------------------------------------


def _list_channels(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channels = [
        {"name": c.get("name", ""), "message_count": len(c.get("messages", []))}
        for c in state.setdefault("channels", [])
    ]
    return {"channels": channels}


def _get_thread(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channel = _find_channel(state, params["channel"])
    root = _find_message(channel, params["message_id"])
    root_id = root.get("thread_id", root.get("id"))
    messages = [
        m
        for m in channel.get("messages", [])
        if m.get("thread_id", m.get("id")) == root_id
    ]
    return {"messages": messages}


def _search_messages(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    query = params["query"]
    if not isinstance(query, str) or not query:
        raise InvalidCommandError(f"query must be a non-empty string, got {query!r}")
    needle = query.lower()
    matches = []
    for channel in state.setdefault("channels", []):
        for message in channel.get("messages", []):
            if needle in str(message.get("text", "")).lower():
                matches.append({"channel": channel.get("name"), "message": message})
    return {"matches": matches}


# --- writes ----------------------------------------------------------------


def _send_message(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channel = _find_channel(state, params["channel"])
    text = _require_text(params)
    messages = channel.setdefault("messages", [])
    message = {
        "id": next_id(messages, "m"),
        "author": "agent",
        "text": text,
        "pinned": False,
    }
    messages.append(message)
    return {"ok": True, "message": message}


def _post_standup_update(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    result = _send_message(state, params)
    message = result["message"]
    message["kind"] = "standup"
    return result


def _pin_message(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channel = _find_channel(state, params["channel"])
    message = _find_message(channel, params["message_id"])
    message["pinned"] = True
    return {"ok": True, "message": message}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="chat", name="list_channels", kind="read", handler=_list_channels),
    ToolSpec(
        app="chat",
        name="get_thread",
        kind="read",
        handler=_get_thread,
        required=("channel", "message_id"),
    ),
    ToolSpec(
        app="chat",
        name="search_messages",
        kind="read",
        handler=_search_messages,
        required=("query",),
    ),
    ToolSpec(
        app="chat",
        name="send_message",
        kind="write",
        handler=_send_message,
        required=("channel", "text"),
    ),
    ToolSpec(
        app="chat",
        name="post_standup_update",
        kind="write",
        handler=_post_standup_update,
        required=("channel", "text"),
    ),
    ToolSpec(
        app="chat",
        name="pin_message",
        kind="write",
        handler=_pin_message,
        required=("channel", "message_id"),
    ),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_chat.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/chat.py tests/test_env_chat.py
git commit -m "Phase 2: chat app tools with tests"
```

---

### Task 4: `env/pager.py` — on-call app

**Files:**
- Create: `src/proactive_bench/env/pager.py`
- Test: `tests/test_env_pager.py`

**Interfaces:**
- Consumes: `ToolSpec`, `ToolExecutionError`, `InvalidCommandError`, `next_id` from Task 1.
- Produces:
  - `initial_state() -> dict[str, Any]` returning `{"alerts": [], "runbooks": [], "deploy_windows": [], "rollback_plans": [], "oncall": {"primary": None, "paged": False, "page_log": []}}`
  - `TOOLS: tuple[ToolSpec, ...]` — 8 tools:
    - reads: `list_alerts()`, `get_alert(id)`, `get_runbook(id)`, `get_deploy_window(id)`
    - writes: `ack_alert(id)`, `page_oncall(alert_id, reason)`, `propose_rollback(service, reason)`, `execute_rollback(plan_id)`
  - Alert shape: `{"id", "severity", "service", "status", "summary"}` where status ∈ `firing|acked|resolved`.
  - Rollback plan shape: `{"id", "service", "reason", "status"}` where status ∈ `proposed|executed`; ids from `next_id(plans, "rb")`.
  - `ack_alert` only works from `firing`; `page_oncall` requires the alert to exist, sets `oncall.paged = True`, appends `{"alert_id", "reason"}` to `oncall.page_log`; `execute_rollback` only works from `proposed`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the simulated pager/on-call app (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import pager
from proactive_bench.env.base import ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in pager.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def seed_state() -> dict:
    return {
        "alerts": [
            {"id": "a1", "severity": "sev2", "service": "api", "status": "firing",
             "summary": "latency spike"},
            {"id": "a2", "severity": "sev4", "service": "web", "status": "acked",
             "summary": "disk usage"},
        ],
        "runbooks": [{"id": "rb-latency", "title": "Latency", "steps": ["check metrics"]}],
        "deploy_windows": [
            {"id": "dw1", "start": "2026-11-10T18:00:00Z", "end": "2026-11-10T20:00:00Z",
             "frozen": False}
        ],
        "rollback_plans": [
            {"id": "rb1", "service": "api", "reason": "bad deploy", "status": "proposed"}
        ],
        "oncall": {"primary": "sam", "paged": False, "page_log": []},
    }


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in pager.TOOLS if s.kind == "read"}
    writes = {s.name for s in pager.TOOLS if s.kind == "write"}
    assert reads == {"list_alerts", "get_alert", "get_runbook", "get_deploy_window"}
    assert writes == {"ack_alert", "page_oncall", "propose_rollback", "execute_rollback"}


def test_initial_state_is_fresh_each_call() -> None:
    a, b = pager.initial_state(), pager.initial_state()
    assert a == {
        "alerts": [],
        "runbooks": [],
        "deploy_windows": [],
        "rollback_plans": [],
        "oncall": {"primary": None, "paged": False, "page_log": []},
    }
    a["alerts"].append(1)
    assert b["alerts"] == []


def test_list_and_get_alerts() -> None:
    state = seed_state()
    assert len(get("list_alerts").handler(state, {})["alerts"]) == 2
    assert get("get_alert").handler(state, {"id": "a1"})["alert"]["severity"] == "sev2"
    with pytest.raises(ToolExecutionError):
        get("get_alert").handler(state, {"id": "a9"})


def test_get_runbook_and_deploy_window() -> None:
    state = seed_state()
    assert get("get_runbook").handler(state, {"id": "rb-latency"})["runbook"]["title"] == "Latency"
    window = get("get_deploy_window").handler(state, {"id": "dw1"})["window"]
    assert window["frozen"] is False
    with pytest.raises(ToolExecutionError):
        get("get_deploy_window").handler(state, {"id": "nope"})


def test_ack_alert_only_from_firing() -> None:
    state = seed_state()
    get("ack_alert").handler(state, {"id": "a1"})
    assert state["alerts"][0]["status"] == "acked"
    with pytest.raises(ToolExecutionError, match="not 'firing'"):
        get("ack_alert").handler(state, {"id": "a1"})
    with pytest.raises(ToolExecutionError, match="not 'firing'"):
        get("ack_alert").handler(state, {"id": "a2"})


def test_page_oncall_logs_and_sets_paged() -> None:
    state = seed_state()
    result = get("page_oncall").handler(state, {"alert_id": "a1", "reason": "customer impact"})
    assert result["ok"] is True
    assert state["oncall"]["paged"] is True
    assert state["oncall"]["page_log"] == [{"alert_id": "a1", "reason": "customer impact"}]
    with pytest.raises(ToolExecutionError):
        get("page_oncall").handler(state, {"alert_id": "a9", "reason": "x"})


def test_propose_and_execute_rollback_lifecycle() -> None:
    state = seed_state()
    result = get("propose_rollback").handler(state, {"service": "web", "reason": "css break"})
    plan = result["plan"]
    assert plan["id"] == "rb2"
    assert plan["status"] == "proposed"
    get("execute_rollback").handler(state, {"plan_id": "rb2"})
    assert state["rollback_plans"][1]["status"] == "executed"
    with pytest.raises(ToolExecutionError, match="not 'proposed'"):
        get("execute_rollback").handler(state, {"plan_id": "rb2"})
    with pytest.raises(ToolExecutionError, match="no rollback plan"):
        get("execute_rollback").handler(state, {"plan_id": "rb1x"})


def test_execute_rollback_requires_existing_proposed_plan() -> None:
    state = seed_state()
    get("execute_rollback").handler(state, {"plan_id": "rb1"})
    assert state["rollback_plans"][0]["status"] == "executed"
    with pytest.raises(ToolExecutionError):
        get("execute_rollback").handler(state, {"plan_id": "rb1"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_pager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.pager'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Simulated on-call pager app: alerts, runbooks, deploy windows, rollbacks."""

from __future__ import annotations

from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    next_id,
)


def initial_state() -> dict[str, Any]:
    return {
        "alerts": [],
        "runbooks": [],
        "deploy_windows": [],
        "rollback_plans": [],
        "oncall": {"primary": None, "paged": False, "page_log": []},
    }


def _require_id(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _find(collection: list[dict[str, Any]], value: str, label: str) -> dict[str, Any]:
    for item in collection:
        if item.get("id") == value:
            return item
    raise ToolExecutionError(f"no {label} with id {value!r}")


def _find_alert(state: dict[str, Any], alert_id: str) -> dict[str, Any]:
    return _find(state.setdefault("alerts", []), alert_id, "alert")


def _require_text(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


# --- reads -----------------------------------------------------------------


def _list_alerts(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"alerts": list(state.setdefault("alerts", []))}


def _get_alert(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"alert": _find_alert(state, _require_id(params, "id"))}


def _get_runbook(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    runbook = _find(state.setdefault("runbooks", []), _require_id(params, "id"), "runbook")
    return {"runbook": runbook}


def _get_deploy_window(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": _find(
            state.setdefault("deploy_windows", []), _require_id(params, "id"), "deploy window"
        )
    }


# --- writes ----------------------------------------------------------------


def _ack_alert(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    alert = _find_alert(state, _require_id(params, "id"))
    if alert.get("status") != "firing":
        raise ToolExecutionError(
            f"alert {alert['id']} is {alert.get('status')!r}, not 'firing'"
        )
    alert["status"] = "acked"
    return {"ok": True, "alert": alert}


def _page_oncall(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    alert_id = _require_id(params, "alert_id")
    reason = _require_text(params, "reason")
    _find_alert(state, alert_id)
    oncall = state.setdefault("oncall", {"primary": None, "paged": False, "page_log": []})
    oncall["paged"] = True
    oncall.setdefault("page_log", []).append({"alert_id": alert_id, "reason": reason})
    return {"ok": True, "oncall": oncall}


def _propose_rollback(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    service = _require_text(params, "service")
    reason = _require_text(params, "reason")
    plans = state.setdefault("rollback_plans", [])
    plan = {"id": next_id(plans, "rb"), "service": service, "reason": reason, "status": "proposed"}
    plans.append(plan)
    return {"ok": True, "plan": plan}


def _execute_rollback(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    plans = state.setdefault("rollback_plans", [])
    plan = _find(plans, _require_id(params, "plan_id"), "rollback plan")
    if plan.get("status") != "proposed":
        raise ToolExecutionError(
            f"rollback plan {plan['id']} is {plan.get('status')!r}, not 'proposed'"
        )
    plan["status"] = "executed"
    return {"ok": True, "plan": plan}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="pager", name="list_alerts", kind="read", handler=_list_alerts),
    ToolSpec(app="pager", name="get_alert", kind="read", handler=_get_alert, required=("id",)),
    ToolSpec(
        app="pager", name="get_runbook", kind="read", handler=_get_runbook, required=("id",)
    ),
    ToolSpec(
        app="pager",
        name="get_deploy_window",
        kind="read",
        handler=_get_deploy_window,
        required=("id",),
    ),
    ToolSpec(app="pager", name="ack_alert", kind="write", handler=_ack_alert, required=("id",)),
    ToolSpec(
        app="pager",
        name="page_oncall",
        kind="write",
        handler=_page_oncall,
        required=("alert_id", "reason"),
    ),
    ToolSpec(
        app="pager",
        name="propose_rollback",
        kind="write",
        handler=_propose_rollback,
        required=("service", "reason"),
    ),
    ToolSpec(
        app="pager",
        name="execute_rollback",
        kind="write",
        handler=_execute_rollback,
        required=("plan_id",),
    ),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_pager.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/pager.py tests/test_env_pager.py
git commit -m "Phase 2: pager app tools with tests"
```

---

### Task 5: `env/cal.py` — calendar app

**Files:**
- Create: `src/proactive_bench/env/cal.py`
- Test: `tests/test_env_cal.py`

**Interfaces:**
- Consumes: `ToolSpec`, `ToolExecutionError`, `InvalidCommandError`, `next_id`, `as_int` from Task 1.
- Produces:
  - `initial_state() -> dict[str, Any]` returning `{"events": [], "focus_blocks": []}`
  - `TOOLS: tuple[ToolSpec, ...]` — 7 tools:
    - reads: `list_events()`, `get_event(id)`, `get_free_slots(date)` (optional int `duration_minutes`, default 60)
    - writes: `create_event(title, start, end)` (optional list `attendees`), `decline_event(id)`, `propose_reschedule(id, start, end)`, `block_focus_time(title, start, end)`
  - Event shape: `{"id", "title", "start", "end", "attendees", "status"}` where status ∈ `confirmed|declined|reschedule_proposed`; reschedule proposals add `proposed_start`/`proposed_end`. Focus block shape: `{"id", "title", "start", "end"}`.
  - Timestamps are ISO-8601 strings; naive timestamps are treated as UTC; `end` must be strictly after `start` or the tool raises `InvalidCommandError`.
  - Event ids from `next_id(events, "e")`, focus blocks from `next_id(blocks, "f")`.
  - `get_free_slots(date)`: workday 09:00–17:00 UTC on `date`, candidate starts every 30 minutes, slot length `duration_minutes` (default 60), excluding overlaps with non-declined events and focus blocks; returns `{"slots": [{"start", "end"}]}` with `+00:00` ISO strings.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the simulated calendar app (offline)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from proactive_bench.env import cal
from proactive_bench.env.base import InvalidCommandError, ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in cal.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def seed_state() -> dict:
    return {
        "events": [
            {"id": "e1", "title": "sync", "start": "2026-11-10T10:00:00+00:00",
             "end": "2026-11-10T11:00:00+00:00", "attendees": ["alice"],
             "status": "confirmed"},
            {"id": "e2", "title": "dental", "start": "2026-11-11T14:00:00+00:00",
             "end": "2026-11-11T15:00:00+00:00", "attendees": [],
             "status": "declined"},
        ],
        "focus_blocks": [
            {"id": "f1", "title": "deep work", "start": "2026-11-10T13:00:00+00:00",
             "end": "2026-11-10T14:00:00+00:00"},
        ],
    }


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in cal.TOOLS if s.kind == "read"}
    writes = {s.name for s in cal.TOOLS if s.kind == "write"}
    assert reads == {"list_events", "get_event", "get_free_slots"}
    assert writes == {"create_event", "decline_event", "propose_reschedule", "block_focus_time"}


def test_initial_state_is_fresh_each_call() -> None:
    a, b = cal.initial_state(), cal.initial_state()
    assert a == {"events": [], "focus_blocks": []}
    a["events"].append(1)
    assert b["events"] == []


def test_list_and_get_events() -> None:
    state = seed_state()
    assert len(get("list_events").handler(state, {})["events"]) == 2
    assert get("get_event").handler(state, {"id": "e1"})["event"]["title"] == "sync"
    with pytest.raises(ToolExecutionError):
        get("get_event").handler(state, {"id": "e9"})


def test_create_event_assigns_id_and_validates_range() -> None:
    state = seed_state()
    result = get("create_event").handler(
        state,
        {"title": "1:1", "start": "2026-11-12T09:00:00Z", "end": "2026-11-12T09:30:00Z",
         "attendees": ["bob"]},
    )
    event = result["event"]
    assert event["id"] == "e3"
    assert event["status"] == "confirmed"
    assert event["start"] == "2026-11-12T09:00:00+00:00"
    with pytest.raises(InvalidCommandError, match="after start"):
        get("create_event").handler(
            state,
            {"title": "bad", "start": "2026-11-12T10:00:00Z", "end": "2026-11-12T09:00:00Z"},
        )
    with pytest.raises(InvalidCommandError, match="ISO-8601"):
        get("create_event").handler(
            state, {"title": "bad", "start": "not-a-time", "end": "2026-11-12T09:00:00Z"}
        )


def test_create_event_rejects_non_list_attendees() -> None:
    with pytest.raises(InvalidCommandError, match="attendees"):
        get("create_event").handler(
            seed_state(),
            {"title": "x", "start": "2026-11-12T09:00:00Z", "end": "2026-11-12T10:00:00Z",
             "attendees": "alice"},
        )


def test_decline_event_transitions_once() -> None:
    state = seed_state()
    get("decline_event").handler(state, {"id": "e1"})
    assert state["events"][0]["status"] == "declined"
    with pytest.raises(ToolExecutionError, match="already declined"):
        get("decline_event").handler(state, {"id": "e1"})


def test_propose_reschedule_sets_proposed_fields() -> None:
    state = seed_state()
    get("propose_reschedule").handler(
        state,
        {"id": "e1", "start": "2026-11-10T15:00:00Z", "end": "2026-11-10T16:00:00Z"},
    )
    event = state["events"][0]
    assert event["status"] == "reschedule_proposed"
    assert event["proposed_start"] == "2026-11-10T15:00:00+00:00"
    assert event["proposed_end"] == "2026-11-10T16:00:00+00:00"
    with pytest.raises(ToolExecutionError, match="declined"):
        get("propose_reschedule").handler(
            state, {"id": "e2", "start": "2026-11-11T16:00:00Z", "end": "2026-11-11T17:00:00Z"}
        )


def test_block_focus_time_appends_block() -> None:
    state = seed_state()
    result = get("block_focus_time").handler(
        state,
        {"title": "write spec", "start": "2026-11-13T09:00:00Z", "end": "2026-11-13T11:00:00Z"},
    )
    assert result["block"]["id"] == "f2"
    assert len(state["focus_blocks"]) == 2


def test_get_free_slots_skips_busy_and_declined() -> None:
    state = seed_state()
    result = get("get_free_slots").handler(state, {"date": "2026-11-10"})
    slots = result["slots"]
    starts = [s["start"] for s in slots]
    # Busy: event 10:00-11:00 blocks 09:30/10:00/10:30 candidates; focus 13:00-14:00
    # blocks 12:30/13:00/13:30 candidates (60-min slots).
    assert "2026-11-10T09:00:00+00:00" in starts
    assert "2026-11-10T11:00:00+00:00" in starts
    assert "2026-11-10T14:00:00+00:00" in starts
    for blocked in (
        "2026-11-10T09:30:00+00:00",
        "2026-11-10T10:00:00+00:00",
        "2026-11-10T10:30:00+00:00",
        "2026-11-10T12:30:00+00:00",
        "2026-11-10T13:00:00+00:00",
        "2026-11-10T13:30:00+00:00",
    ):
        assert blocked not in starts
    assert starts[0] == "2026-11-10T09:00:00+00:00"
    assert starts[-1] == "2026-11-10T16:00:00+00:00"
    # 15 candidates in 09:00..16:00 step 30, minus 6 blocked = 9.
    assert len(slots) == 9
    # Declined event on 11-11 does not block that day.
    day2 = get("get_free_slots").handler(state, {"date": "2026-11-11"})["slots"]
    assert len(day2) == 15


def test_get_free_slots_honors_duration_and_validates_date() -> None:
    state = seed_state()
    slots = get("get_free_slots").handler(
        state, {"date": "2026-11-10", "duration_minutes": 120}
    )["slots"]
    for slot in slots:
        assert datetime.fromisoformat(slot["end"]) - datetime.fromisoformat(slot["start"]) == (
            timedelta(minutes=120)
        )
    with pytest.raises(InvalidCommandError, match="YYYY-MM-DD"):
        get("get_free_slots").handler(state, {"date": "10-11-2026"})
    with pytest.raises(InvalidCommandError, match="duration_minutes"):
        get("get_free_slots").handler(state, {"date": "2026-11-10", "duration_minutes": 0})
```

- [ ] **Step: Run test to verify it fails**

Run: `pytest tests/test_env_cal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.cal'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Simulated calendar app: events, declines, reschedule proposals, focus blocks."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    as_int,
    next_id,
)

_WORKDAY_START_HOUR = 9
_WORKDAY_END_HOUR = 17
_SLOT_STEP_MINUTES = 30


def initial_state() -> dict[str, Any]:
    return {"events": [], "focus_blocks": []}


def _parse_ts(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{name} must be an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidCommandError(f"{name} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _try_parse_ts(value: Any) -> datetime | None:
    try:
        return _parse_ts(value, "timestamp")
    except InvalidCommandError:
        return None


def _require_range(params: dict[str, Any]) -> tuple[datetime, datetime]:
    start = _parse_ts(params["start"], "start")
    end = _parse_ts(params["end"], "end")
    if end <= start:
        raise InvalidCommandError(f"end must be after start ({params['start']}..{params['end']})")
    return start, end


def _require_text(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _find_event(state: dict[str, Any], event_id: str) -> dict[str, Any]:
    for event in state.setdefault("events", []):
        if event.get("id") == event_id:
            return event
    raise ToolExecutionError(f"no event with id {event_id!r}")


def _require_id(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


# --- reads -----------------------------------------------------------------


def _list_events(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"events": list(state.setdefault("events", []))}


def _get_event(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"event": _find_event(state, _require_id(params, "id"))}


def _get_free_slots(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    date = params["date"]
    if not isinstance(date, str):
        raise InvalidCommandError(f"date must be YYYY-MM-DD, got {date!r}")
    try:
        day = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise InvalidCommandError(f"date must be YYYY-MM-DD, got {date!r}") from exc
    duration = as_int(params.get("duration_minutes", 60), "duration_minutes")
    if duration <= 0:
        raise InvalidCommandError(f"duration_minutes must be positive, got {duration}")
    if duration > 1440:
        raise InvalidCommandError(f"duration_minutes must be at most 1440, got {duration}")
    day_start = datetime.combine(day, time(hour=_WORKDAY_START_HOUR), tzinfo=UTC)
    day_end = datetime.combine(day, time(hour=_WORKDAY_END_HOUR), tzinfo=UTC)
    busy: list[tuple[datetime, datetime]] = []
    for item in [*state.setdefault("events", []), *state.setdefault("focus_blocks", [])]:
        if item.get("status") == "declined":
            continue
        start = _try_parse_ts(item.get("start"))
        end = _try_parse_ts(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        if start < day_end and end > day_start:
            busy.append((max(start, day_start), min(end, day_end)))
    slot_delta = timedelta(minutes=duration)
    step = timedelta(minutes=_SLOT_STEP_MINUTES)
    slots: list[dict[str, str]] = []
    cursor = day_start
    while cursor + slot_delta <= day_end:
        slot_end = cursor + slot_delta
        if all(slot_end <= b0 or cursor >= b1 for b0, b1 in busy):
            slots.append({"start": cursor.isoformat(), "end": slot_end.isoformat()})
        cursor += step
    return {"slots": slots}


# --- writes ----------------------------------------------------------------


def _create_event(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    title = _require_text(params, "title")
    start, end = _require_range(params)
    attendees = params.get("attendees", [])
    if not isinstance(attendees, list):
        raise InvalidCommandError(f"attendees must be a list, got {attendees!r}")
    events = state.setdefault("events", [])
    event = {
        "id": next_id(events, "e"),
        "title": title,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "attendees": list(attendees),
        "status": "confirmed",
    }
    events.append(event)
    return {"ok": True, "event": event}


def _decline_event(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    event = _find_event(state, _require_id(params, "id"))
    if event.get("status") == "declined":
        raise ToolExecutionError(f"event {event['id']} already declined")
    event["status"] = "declined"
    return {"ok": True, "event": event}


def _propose_reschedule(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    event = _find_event(state, _require_id(params, "id"))
    if event.get("status") == "declined":
        raise ToolExecutionError(f"event {event['id']} is declined")
    start, end = _require_range(params)
    event["status"] = "reschedule_proposed"
    event["proposed_start"] = start.isoformat()
    event["proposed_end"] = end.isoformat()
    return {"ok": True, "event": event}


def _block_focus_time(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    title = _require_text(params, "title")
    start, end = _require_range(params)
    blocks = state.setdefault("focus_blocks", [])
    block = {"id": next_id(blocks, "f"), "title": title, "start": start.isoformat(),
             "end": end.isoformat()}
    blocks.append(block)
    return {"ok": True, "block": block}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="cal", name="list_events", kind="read", handler=_list_events),
    ToolSpec(app="cal", name="get_event", kind="read", handler=_get_event, required=("id",)),
    ToolSpec(
        app="cal", name="get_free_slots", kind="read", handler=_get_free_slots, required=("date",)
    ),
    ToolSpec(
        app="cal",
        name="create_event",
        kind="write",
        handler=_create_event,
        required=("title", "start", "end"),
    ),
    ToolSpec(
        app="cal", name="decline_event", kind="write", handler=_decline_event, required=("id",)
    ),
    ToolSpec(
        app="cal",
        name="propose_reschedule",
        kind="write",
        handler=_propose_reschedule,
        required=("id", "start", "end"),
    ),
    ToolSpec(
        app="cal",
        name="block_focus_time",
        kind="write",
        handler=_block_focus_time,
        required=("title", "start", "end"),
    ),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_cal.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/cal.py tests/test_env_cal.py
git commit -m "Phase 2: cal app tools with tests"
```

---

### Task 6: `env/registry.py` + `env/parser.py` — allowlist and command parsing

**Files:**
- Create: `src/proactive_bench/env/registry.py`
- Create: `src/proactive_bench/env/parser.py`
- Test: `tests/test_env_registry.py`, `tests/test_env_parser.py`

**Interfaces:**
- Consumes: `TOOLS` + `initial_state` from the four app modules (Tasks 2–5); error types + `ToolSpec` from Task 1.
- Produces (relied on by Tasks 7–9):
  - `registry.APPS: dict[str, tuple[ToolSpec, ...]]` with keys exactly `repo`, `chat`, `pager`, `cal`
  - `registry.INITIAL_STATES: dict[str, Callable[[], dict[str, Any]]]`
  - `registry.get_tool(app: str, name: str) -> ToolSpec` — raises `UnknownAppError` / `UnknownToolError`
  - `registry.all_tools() -> tuple[ToolSpec, ...]` (32 tools, app order repo, chat, pager, cal)
  - `registry.tool_kind(app: str, name: str) -> ToolKind`
  - `parser.Command` frozen dataclass `(app: str, function: str, parameter: dict[str, Any])`
  - `parser.parse_command(raw: Any) -> Command` — accepts extra keys (e.g. `event_id`, `time`), defaults missing `parameter` to `{}`, rejects `null`/non-object `parameter`, deep-copies parameters, validates app/function/required params via the registry
  - `parser.require_kind(command: Command, kind: ToolKind) -> ToolSpec` — raises `ToolKindError` on mismatch

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the tool registry (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import registry
from proactive_bench.env.base import UnknownAppError, UnknownToolError

EXPECTED_TOOLS = {
    "repo": {
        "read": {"list_prs", "get_pr", "get_checks", "list_issues", "get_issue"},
        "write": {"post_comment", "request_review", "retrigger_ci", "open_draft_pr",
                  "merge_pr", "revert_commit"},
    },
    "chat": {
        "read": {"list_channels", "get_thread", "search_messages"},
        "write": {"send_message", "post_standup_update", "pin_message"},
    },
    "pager": {
        "read": {"list_alerts", "get_alert", "get_runbook", "get_deploy_window"},
        "write": {"ack_alert", "page_oncall", "propose_rollback", "execute_rollback"},
    },
    "cal": {
        "read": {"list_events", "get_event", "get_free_slots"},
        "write": {"create_event", "decline_event", "propose_reschedule", "block_focus_time"},
    },
}


def test_apps_match_spec_table_exactly() -> None:
    assert set(registry.APPS) == set(EXPECTED_TOOLS)
    for app, expected in EXPECTED_TOOLS.items():
        specs = registry.APPS[app]
        for kind in ("read", "write"):
            actual = {s.name for s in specs if s.kind == kind}
            assert actual == expected[kind], f"{app} {kind}"


def test_total_tool_count_is_32() -> None:
    assert len(registry.all_tools()) == 32


def test_initial_states_cover_all_apps() -> None:
    assert set(registry.INITIAL_STATES) == set(registry.APPS)
    for factory in registry.INITIAL_STATES.values():
        state = factory()
        assert isinstance(state, dict)
        assert factory() is not state


def test_get_tool_returns_spec() -> None:
    spec = registry.get_tool("repo", "merge_pr")
    assert spec.kind == "write"
    assert spec.required == ("number",)
    assert spec.full_name == "repo.merge_pr"


def test_get_tool_unknown_app_and_tool() -> None:
    with pytest.raises(UnknownAppError, match="slack"):
        registry.get_tool("slack", "post")
    with pytest.raises(UnknownToolError, match="repo.delete_pr"):
        registry.get_tool("repo", "delete_pr")


def test_tool_kind_delegates() -> None:
    assert registry.tool_kind("chat", "send_message") == "write"
    assert registry.tool_kind("cal", "list_events") == "read"


def test_function_names_are_globally_unique() -> None:
    names = [t.name for t in registry.all_tools()]
    assert len(names) == len(set(names))
```

```python
"""Tests for the command parser (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env.base import (
    InvalidCommandError,
    MissingParameterError,
    ToolKindError,
    UnknownAppError,
    UnknownToolError,
)
from proactive_bench.env.parser import Command, parse_command, require_kind


def test_parse_command_happy_path() -> None:
    command = parse_command(
        {"app": "repo", "function": "get_pr", "parameter": {"number": 1}}
    )
    assert command == Command(app="repo", function="get_pr", parameter={"number": 1})


def test_parse_command_defaults_missing_parameter() -> None:
    command = parse_command({"app": "repo", "function": "list_prs"})
    assert command.parameter == {}


def test_parse_command_ignores_extra_keys() -> None:
    command = parse_command(
        {
            "event_id": "e1",
            "time": "2026-11-10T09:00:00Z",
            "app": "chat",
            "function": "send_message",
            "parameter": {"channel": "eng", "text": "hi"},
        }
    )
    assert command.app == "chat"
    assert command.function == "send_message"
    assert "event_id" not in vars(command)


def test_parse_command_deep_copies_parameter() -> None:
    raw = {"app": "repo", "function": "post_comment", "parameter": {"number": 1, "body": "x"}}
    command = parse_command(raw)
    raw["parameter"]["body"] = "mutated"
    assert command.parameter["body"] == "x"


def test_parse_command_rejects_structural_errors() -> None:
    for bad in (
        "get_pr",
        42,
        None,
        [],
        {"function": "get_pr"},
        {"app": "repo"},
        {"app": 7, "function": "get_pr"},
        {"app": "repo", "function": ""},
        {"app": "repo", "function": "get_pr", "parameter": None},
        {"app": "repo", "function": "get_pr", "parameter": [1]},
    ):
        with pytest.raises(InvalidCommandError):
            parse_command(bad)


def test_parse_command_unknown_app_or_tool() -> None:
    with pytest.raises(UnknownAppError):
        parse_command({"app": "slack", "function": "post", "parameter": {}})
    with pytest.raises(UnknownToolError):
        parse_command({"app": "repo", "function": "delete_pr", "parameter": {}})


def test_parse_command_missing_required_param() -> None:
    with pytest.raises(MissingParameterError, match="number"):
        parse_command({"app": "repo", "function": "get_pr", "parameter": {}})
    with pytest.raises(MissingParameterError, match="channel, text"):
        parse_command({"app": "chat", "function": "send_message", "parameter": {}})


def test_require_kind_ok_and_mismatch() -> None:
    read_cmd = parse_command({"app": "repo", "function": "list_prs"})
    spec = require_kind(read_cmd, "read")
    assert spec.name == "list_prs"
    with pytest.raises(ToolKindError, match="read tool, not write"):
        require_kind(read_cmd, "write")
    write_cmd = parse_command(
        {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
    )
    assert require_kind(write_cmd, "write").kind == "write"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_registry.py tests/test_env_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Allowlist registry aggregating all four simulated apps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from proactive_bench.env import cal, chat, pager, repo
from proactive_bench.env.base import ToolKind, ToolSpec, UnknownAppError, UnknownToolError

APPS: dict[str, tuple[ToolSpec, ...]] = {
    "repo": repo.TOOLS,
    "chat": chat.TOOLS,
    "pager": pager.TOOLS,
    "cal": cal.TOOLS,
}

INITIAL_STATES: dict[str, Callable[[], dict[str, Any]]] = {
    "repo": repo.initial_state,
    "chat": chat.initial_state,
    "pager": pager.initial_state,
    "cal": cal.initial_state,
}


def get_tool(app: str, name: str) -> ToolSpec:
    """Look up one allowlisted tool; raise ``UnknownAppError``/``UnknownToolError``."""
    if app not in APPS:
        known = ", ".join(sorted(APPS))
        raise UnknownAppError(f"unknown app {app!r} (known: {known})")
    for spec in APPS[app]:
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in APPS[app])
    raise UnknownToolError(f"unknown tool {app}.{name} (known: {known})")


def all_tools() -> tuple[ToolSpec, ...]:
    return tuple(spec for specs in APPS.values() for spec in specs)


def tool_kind(app: str, name: str) -> ToolKind:
    return get_tool(app, name).kind
```

```python
"""Parse raw agent/event command dicts into validated Commands."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolKind,
    ToolKindError,
    ToolSpec,
)
from proactive_bench.env.registry import get_tool


@dataclass(frozen=True)
class Command:
    """One validated tool invocation: app id, function name, parameters."""

    app: str
    function: str
    parameter: dict[str, Any]


def parse_command(raw: Any) -> Command:
    """Validate a raw ``{app, function, parameter}`` mapping into a ``Command``.

    Extra keys (e.g. scenario ``event_id``/``time``) are ignored. A missing
    ``parameter`` defaults to ``{}``; ``null`` or non-object parameters raise.
    """
    if not isinstance(raw, Mapping):
        raise InvalidCommandError(
            f"command must be an object, got {type(raw).__name__}"
        )
    app = raw.get("app")
    function = raw.get("function")
    if not isinstance(app, str) or not app:
        raise InvalidCommandError(f"command.app must be a non-empty string, got {app!r}")
    if not isinstance(function, str) or not function:
        raise InvalidCommandError(
            f"command.function must be a non-empty string, got {function!r}"
        )
    params = raw.get("parameter", {})
    if not isinstance(params, Mapping):
        raise InvalidCommandError(
            f"command.parameter must be an object, got {type(params).__name__}"
        )
    spec = get_tool(app, function)
    copied = deepcopy(dict(params))
    spec.check_params(copied)
    return Command(app=app, function=function, parameter=copied)


def require_kind(command: Command, kind: ToolKind) -> ToolSpec:
    """Return the tool spec if ``command`` matches ``kind``, else raise."""
    spec = get_tool(command.app, command.function)
    if spec.kind != kind:
        raise ToolKindError(
            f"{spec.full_name} is a {spec.kind} tool, not {kind}"
        )
    return spec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_env_registry.py tests/test_env_parser.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/registry.py src/proactive_bench/env/parser.py tests/test_env_registry.py tests/test_env_parser.py
git commit -m "Phase 2: tool registry and command parser"
```

---

### Task 7: `env/workspace.py` — checkpoint isolation and execution

**Files:**
- Create: `src/proactive_bench/env/workspace.py`
- Test: `tests/test_env_workspace.py`

**Interfaces:**
- Consumes: `registry.APPS`, `registry.INITIAL_STATES`, `registry.get_tool`, `parser.Command` (Tasks 6–7), `deepcopy` from stdlib.
- Produces (used by Phase 4/6 runner and Task 9):
  - `class Workspace`
    - `__init__(states: Mapping[str, Mapping[str, Any]] | None = None)` — deep-copies provided per-app states; fills unprovided apps from `INITIAL_STATES`; raises `UnknownAppError` for unknown app keys
    - `clone() -> Workspace` — fresh deep copy (isolation unit for checkpoints)
    - `state(app: str) -> dict[str, Any]` — live view of one app's state (inspect only; mutating it is the caller's responsibility)
    - `snapshot() -> dict[str, dict[str, Any]]` — deep copy of all states (safe for serialization)
    - `execute(command: Command) -> dict[str, Any]` — look up tool, `check_params`, run handler, return a **deep copy** of the handler result so results never alias live state

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for Workspace isolation and execution (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env.base import MissingParameterError, UnknownAppError, UnknownToolError
from proactive_bench.env.parser import Command, parse_command
from proactive_bench.env.workspace import Workspace


def test_default_workspace_has_all_four_apps() -> None:
    ws = Workspace()
    assert set(ws.snapshot()) == {"repo", "chat", "pager", "cal"}
    assert ws.snapshot()["repo"] == {"prs": [], "issues": [], "commits": []}


def test_unknown_app_key_rejected() -> None:
    with pytest.raises(UnknownAppError):
        Workspace({"slack": {}})


def test_constructor_deep_copies_caller_state() -> None:
    original = {"prs": [{"number": 1}], "issues": [], "commits": []}
    ws = Workspace({"repo": original})
    original["prs"].append({"number": 2})
    assert len(ws.state("repo")["prs"]) == 1


def test_clone_is_isolated_from_parent() -> None:
    ws = Workspace()
    clone = ws.clone()
    clone.state("repo").setdefault("prs", []).append({"number": 1})
    assert ws.state("repo")["prs"] == []
    ws.state("chat").setdefault("channels", []).append({"name": "eng", "messages": []})
    assert clone.state("chat")["channels"] == []


def test_execute_read_and_write_with_result_isolation() -> None:
    ws = Workspace(
        {"chat": {"channels": [{"name": "eng", "messages": []}]}}
    )
    write = ws.execute(
        parse_command(
            {"app": "chat", "function": "send_message",
             "parameter": {"channel": "eng", "text": "hello"}}
        )
    )
    assert write["ok"] is True
    message = write["message"]
    message["text"] = "mutated-by-caller"
    assert ws.state("chat")["channels"][0]["messages"][0]["text"] == "hello"
    read = ws.execute(parse_command({"app": "chat", "function": "list_channels"}))
    assert read["channels"] == [{"name": "eng", "message_count": 1}]


def test_execute_validates_tool_and_params() -> None:
    ws = Workspace()
    with pytest.raises(UnknownToolError):
        ws.execute(Command(app="repo", function="nope", parameter={}))
    with pytest.raises(MissingParameterError):
        ws.execute(Command(app="repo", function="get_pr", parameter={}))


def test_snapshot_is_deep_copy() -> None:
    ws = Workspace()
    snap = ws.snapshot()
    snap["repo"]["prs"].append("x")
    assert ws.state("repo")["prs"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_workspace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.workspace'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Checkpoint workspace: deep-copy isolation plus validated tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from proactive_bench.env.base import ToolError, UnknownAppError
from proactive_bench.env.parser import Command
from proactive_bench.env.registry import APPS, INITIAL_STATES, get_tool


class Workspace:
    """Holds per-app state for one checkpoint.

    ``state()`` returns a live view (inspection/pre-seeding only); ``clone()``
    and ``snapshot()`` return deep copies. ``execute()`` never lets a handler
    result alias live state.
    """

    def __init__(self, states: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        provided = dict(states or {})
        for app in provided:
            if app not in APPS:
                known = ", ".join(sorted(APPS))
                raise UnknownAppError(f"unknown app {app!r} (known: {known})")
        self._states: dict[str, dict[str, Any]] = {}
        for app in APPS:
            if app in provided:
                self._states[app] = deepcopy(dict(provided[app]))
            else:
                self._states[app] = INITIAL_STATES[app]()

    def clone(self) -> Workspace:
        return Workspace(self._states)

    def state(self, app: str) -> dict[str, Any]:
        if app not in self._states:
            known = ", ".join(sorted(self._states))
            raise UnknownAppError(f"unknown app {app!r} (known: {known})")
        return self._states[app]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._states)

    def execute(self, command: Command) -> dict[str, Any]:
        spec = get_tool(command.app, command.function)
        spec.check_params(command.parameter)
        result = spec.handler(self.state(command.app), command.parameter)
        if not isinstance(result, dict):
            raise ToolError(
                f"{spec.full_name} handler returned {type(result).__name__}, expected dict"
            )
        return deepcopy(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_workspace.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/workspace.py tests/test_env_workspace.py
git commit -m "Phase 2: workspace isolation and execution"
```

---

### Task 8: `env/classifier.py` — deterministic L0–L3 classifier

**Files:**
- Create: `src/proactive_bench/env/classifier.py`
- Test: `tests/test_env_classifier.py`

**Interfaces:**
- Consumes: `registry.get_tool`, `registry.all_tools`, `parser.Command`, error types (Tasks 1, 6, 7).
- Produces (used by Phase 6 runner + Phase 8 report):
  - `ActionLevel = Literal["L0", "L1", "L2", "L3"]`
  - `classify_action_level(action: Command | Mapping[str, Any] | None) -> ActionLevel`
    - `None` → `"L0"`
    - `Command` → resolve via `get_tool(action.app, action.function)`
    - `Mapping` → requires string `app` and `function` keys, resolved via `get_tool`
    - read tool → `"L0"` (SPEC §3.4: L0 includes read-only observation)
    - write tool → `_LEVELS[function]`; write tool missing from `_LEVELS` raises `ToolError`
    - unknown app/tool → `UnknownAppError`/`UnknownToolError`; non-command input → `InvalidCommandError`
  - `_LEVELS` table must contain exactly the 17 write tools listed in Global Constraints.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the deterministic action-level classifier (offline)."""

from __future__ import annotations

from typing import get_args

import pytest

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolError,
    UnknownAppError,
    UnknownToolError,
)
from proactive_bench.env.classifier import ActionLevel, classify_action_level
from proactive_bench.env.parser import Command, parse_command
from proactive_bench.env.registry import all_tools

EXPECTED_LEVELS = {
    # L1 — notify / inform
    "post_comment": "L1",
    "request_review": "L1",
    "retrigger_ci": "L1",
    "send_message": "L1",
    "post_standup_update": "L1",
    "pin_message": "L1",
    "ack_alert": "L1",
    "page_oncall": "L1",
    # L2 — prepare / draft
    "open_draft_pr": "L2",
    "propose_rollback": "L2",
    "propose_reschedule": "L2",
    "block_focus_time": "L2",
    # L3 — execute
    "merge_pr": "L3",
    "revert_commit": "L3",
    "execute_rollback": "L3",
    "create_event": "L3",
    "decline_event": "L3",
}


def test_none_is_l0() -> None:
    assert classify_action_level(None) == "L0"


def test_every_read_tool_is_l0() -> None:
    for spec in all_tools():
        if spec.kind == "read":
            command = Command(app=spec.app, function=spec.name, parameter={})
            assert classify_action_level(command) == "L0", spec.full_name


def test_every_write_tool_has_expected_level() -> None:
    writes = [s for s in all_tools() if s.kind == "write"]
    assert len(writes) == 17
    for spec in writes:
        assert spec.name in EXPECTED_LEVELS, f"missing level for {spec.full_name}"
        command = Command(app=spec.app, function=spec.name, parameter={})
        level = classify_action_level(command)
        assert level == EXPECTED_LEVELS[spec.name], spec.full_name
        assert level in ("L1", "L2", "L3")


def test_command_and_mapping_forms_agree() -> None:
    raw = {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
    command = parse_command(raw)
    assert classify_action_level(command) == classify_action_level(raw) == "L3"


def test_unknown_tool_and_app_raise() -> None:
    with pytest.raises(UnknownToolError):
        classify_action_level({"app": "repo", "function": "delete_pr"})
    with pytest.raises(UnknownAppError):
        classify_action_level({"app": "slack", "function": "post"})


def test_malformed_mapping_raises() -> None:
    for bad in (42, "merge_pr", {"function": "merge_pr"}, {"app": "repo"}):
        with pytest.raises(InvalidCommandError):
            classify_action_level(bad)  # type: ignore[arg-type]


def test_deterministic_across_calls() -> None:
    action = {"app": "pager", "function": "execute_rollback", "parameter": {"plan_id": "rb1"}}
    assert classify_action_level(action) == classify_action_level(action) == "L3"


def test_action_level_type_is_literal() -> None:
    assert set(get_args(ActionLevel)) == {"L0", "L1", "L2", "L3"}


def test_internal_levels_table_is_complete() -> None:
    from proactive_bench.env import classifier as _c

    write_names = {s.name for s in all_tools() if s.kind == "write"}
    assert set(_c._LEVELS) == write_names


def test_write_tool_missing_from_table_raises_tool_error() -> None:
    from proactive_bench.env import classifier as _c

    with pytest.raises(ToolError, match="no action-level mapping"):
        _c._LEVELS.pop("merge_pr")
        try:
            classify_action_level(
                {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
            )
        finally:
            _c._LEVELS["merge_pr"] = "L3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.env.classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Deterministic action-level classifier (SPEC §3.4): final tool + args → L0–L3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, get_args

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolError,
)
from proactive_bench.env.parser import Command
from proactive_bench.env.registry import get_tool

ActionLevel = Literal["L0", "L1", "L2", "L3"]

_LEVELS: dict[str, ActionLevel] = {
    "post_comment": "L1",
    "request_review": "L1",
    "retrigger_ci": "L1",
    "open_draft_pr": "L2",
    "merge_pr": "L3",
    "revert_commit": "L3",
    "send_message": "L1",
    "post_standup_update": "L1",
    "pin_message": "L1",
    "ack_alert": "L1",
    "page_oncall": "L1",
    "propose_rollback": "L2",
    "execute_rollback": "L3",
    "create_event": "L3",
    "decline_event": "L3",
    "propose_reschedule": "L2",
    "block_focus_time": "L2",
}


def classify_action_level(action: Command | Mapping[str, Any] | None) -> ActionLevel:
    """Map a final action to L0–L3.

    ``None`` (do nothing) and read-only tools are L0. Write tools use the
    deterministic ``_LEVELS`` table keyed by tool name; arguments are part of
    the input contract but no argument-dependent rules are active in v1.
    Raises ``UnknownAppError``/``UnknownToolError`` for unregistered tools.
    """
    if action is None:
        return "L0"
    if isinstance(action, Command):
        spec = get_tool(action.app, action.function)
    elif isinstance(action, Mapping):
        app = action.get("app")
        function = action.get("function")
        if not isinstance(app, str) or not isinstance(function, str):
            raise InvalidCommandError(
                "action must include string 'app' and 'function' keys"
            )
        spec = get_tool(app, function)
    else:
        raise InvalidCommandError(
            f"action must be a command or null, got {type(action).__name__}"
        )
    if spec.kind == "read":
        return "L0"
    level = _LEVELS.get(spec.name)
    if level is None:
        raise ToolError(
            f"write tool {spec.full_name} has no action-level mapping "
            "(add it to classifier._LEVELS)"
        )
    return level


def known_levels() -> tuple[ActionLevel, ...]:
    """All action levels, in order (reporting helper)."""
    return get_args(ActionLevel)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_classifier.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/env/classifier.py tests/test_env_classifier.py
git commit -m "Phase 2: action-level classifier L0-L3"
```

---

### Task 9: integration trace, public exports, Phase 2 status bump

**Files:**
- Create: `tests/test_env_trace.py`
- Modify: `src/proactive_bench/env/__init__.py`
- Modify: `tests/test_smoke.py`
- Modify: `README.md` (badge line 8, roadmap rows 99–100)
- Modify: `src/proactive_bench/cli.py:37` (phase string)

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: `proactive_bench.env` public API; Phase 2 exit criteria (scripted tool trace executes); README/CLI status at Phase 2.

- [ ] **Step 1: Write the failing integration test**

```python
"""Integration: a scripted multi-app tool trace executes end-to-end (Phase 2 exit)."""

from __future__ import annotations

import pytest

from proactive_bench.env import Workspace, classify_action_level, parse_command
from proactive_bench.env.base import ToolExecutionError


def make_workspace() -> Workspace:
    return Workspace(
        {
            "repo": {
                "prs": [
                    {
                        "number": 1,
                        "title": "fix flaky test",
                        "branch": "fix-flaky",
                        "base": "main",
                        "draft": False,
                        "status": "open",
                        "checks": [{"name": "ci", "status": "fail"}],
                        "reviewers": [],
                        "comments": [],
                        "author": "dev",
                    }
                ],
                "issues": [],
                "commits": [{"sha": "abc123", "message": "wip", "reverted": False}],
            },
            "chat": {"channels": [{"name": "eng", "messages": []}]},
            "pager": {
                "alerts": [
                    {"id": "a1", "severity": "sev2", "service": "api",
                     "status": "firing", "summary": "latency spike"}
                ],
                "runbooks": [],
                "deploy_windows": [],
                "rollback_plans": [],
                "oncall": {"primary": "sam", "paged": False, "page_log": []},
            },
            "cal": {"events": [], "focus_blocks": []},
        }
    )


def test_scripted_tool_trace_executes() -> None:
    ws = make_workspace()
    results: list[dict] = []

    steps = [
        {"app": "repo", "function": "list_prs", "parameter": {}},
        {"app": "repo", "function": "get_checks", "parameter": {"number": 1}},
        {"app": "repo", "function": "retrigger_ci", "parameter": {"number": 1}},
        {"app": "chat", "function": "send_message",
         "parameter": {"channel": "eng", "text": "CI re-running"}},
        {"app": "pager", "function": "ack_alert", "parameter": {"id": "a1"}},
        {"app": "pager", "function": "page_oncall",
         "parameter": {"alert_id": "a1", "reason": "customer impact"}},
        {"app": "cal", "function": "create_event",
         "parameter": {"title": "postmortem", "start": "2026-11-12T16:00:00Z",
                       "end": "2026-11-12T17:00:00Z"}},
        {"app": "repo", "function": "revert_commit", "parameter": {"sha": "abc123"}},
    ]
    for raw in steps:
        results.append(ws.execute(parse_command(raw)))

    assert results[0]["prs"][0]["number"] == 1
    assert results[1]["checks"][0]["status"] == "fail"
    assert results[2]["checks"][0]["status"] == "pending"
    assert results[3]["message"]["text"] == "CI re-running"
    assert results[4]["alert"]["status"] == "acked"
    assert ws.state("pager")["oncall"]["paged"] is True
    assert results[6]["event"]["id"] == "e1"
    assert results[7]["commit"]["reverted"] is True

    # Merge while checks are pending must fail (agent-visible error)...
    with pytest.raises(ToolExecutionError, match="failing or pending"):
        ws.execute(parse_command({"app": "repo", "function": "merge_pr",
                                  "parameter": {"number": 1}}))
    # ...then CI passes and the merge succeeds.
    ws.state("repo")["prs"][0]["checks"][0]["status"] = "pass"
    merged = ws.execute(parse_command({"app": "repo", "function": "merge_pr",
                                       "parameter": {"number": 1}}))
    assert merged["pr"]["status"] == "merged"

    snapshot = ws.snapshot()
    assert snapshot["chat"]["channels"][0]["messages"][0]["text"] == "CI re-running"
    assert snapshot["pager"]["oncall"]["page_log"][0]["alert_id"] == "a1"
    assert snapshot["cal"]["events"][0]["status"] == "confirmed"
    assert snapshot["repo"]["commits"][0]["reverted"] is True


def test_trace_actions_classify_to_expected_levels() -> None:
    assert classify_action_level({"app": "repo", "function": "list_prs"}) == "L0"
    assert classify_action_level(
        {"app": "repo", "function": "retrigger_ci", "parameter": {"number": 1}}
    ) == "L1"
    assert classify_action_level(
        {"app": "chat", "function": "send_message",
         "parameter": {"channel": "eng", "text": "x"}}
    ) == "L1"
    assert classify_action_level(
        {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
    ) == "L3"
    assert classify_action_level(None) == "L0"


def test_trace_checkpoint_isolation() -> None:
    ws = make_workspace()
    checkpoint = ws.clone()
    checkpoint.execute(parse_command({"app": "repo", "function": "merge_pr",
                                      "parameter": {"number": 1, "force": True}}))
    checkpoint.execute(parse_command({"app": "chat", "function": "send_message",
                                      "parameter": {"channel": "eng", "text": "merged!"}}))
    assert ws.state("repo")["prs"][0]["status"] == "open"
    assert ws.state("chat")["channels"][0]["messages"] == []
    # Original inputs to Workspace() are also untouched after a full trace.
    assert checkpoint.snapshot()["repo"]["prs"][0]["status"] == "merged"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_trace.py -v`
Expected: FAIL — `ImportError: cannot import name 'classify_action_level' from 'proactive_bench.env'`

- [ ] **Step 3: Write `env/__init__.py` public exports**

```python
"""Simulated apps: repo, chat, pager, cal — plus parser, isolation, classifier."""

from __future__ import annotations

from proactive_bench.env import cal, chat, pager, repo
from proactive_bench.env.base import (
    Handler,
    InvalidCommandError,
    MissingParameterError,
    ToolError,
    ToolExecutionError,
    ToolKind,
    ToolKindError,
    ToolSpec,
    UnknownAppError,
    UnknownToolError,
    as_int,
    next_id,
)
from proactive_bench.env.classifier import ActionLevel, classify_action_level, known_levels
from proactive_bench.env.parser import Command, parse_command, require_kind
from proactive_bench.env.registry import APPS, INITIAL_STATES, all_tools, get_tool, tool_kind
from proactive_bench.env.workspace import Workspace

__all__ = [
    "APPS",
    "ActionLevel",
    "Command",
    "Handler",
    "INITIAL_STATES",
    "InvalidCommandError",
    "MissingParameterError",
    "ToolError",
    "ToolExecutionError",
    "ToolKind",
    "ToolKindError",
    "ToolSpec",
    "UnknownAppError",
    "UnknownToolError",
    "Workspace",
    "all_tools",
    "as_int",
    "cal",
    "chat",
    "classify_action_level",
    "get_tool",
    "known_levels",
    "next_id",
    "pager",
    "parse_command",
    "repo",
    "require_kind",
    "tool_kind",
]
```

- [ ] **Step 4: Add the env public-API smoke test**

Append to `tests/test_smoke.py`:

```python
def test_env_public_api() -> None:
    from proactive_bench import env

    for name in (
        "APPS",
        "ActionLevel",
        "Command",
        "InvalidCommandError",
        "MissingParameterError",
        "ToolError",
        "ToolExecutionError",
        "ToolKindError",
        "ToolSpec",
        "UnknownAppError",
        "UnknownToolError",
        "Workspace",
        "all_tools",
        "classify_action_level",
        "get_tool",
        "parse_command",
        "require_kind",
    ):
        assert hasattr(env, name), name
```

- [ ] **Step 5: Run full verification**

Run: `pytest -v` → all pass (old + new)
Run: `ruff check src tests` → clean

- [ ] **Step 6: Bump Phase 2 status (README + CLI)**

`README.md` line 8:

```markdown
![Status: Phase 2](https://img.shields.io/badge/status-phase%202-yellow)
```

`README.md` roadmap rows (lines 99–100):

```markdown
| 1 | LLM client + mocked tests | ✅ |
| 2 | Simulated apps + parser + L0–L3 | ✅ **current** |
```

`src/proactive_bench/cli.py` line 37:

```python
        f"(see SPEC.md phases; currently Phase 2).",
```

- [ ] **Step 7: Re-run verification and commit**

Run: `pytest` → PASS; `ruff check src tests` → clean

```bash
git add src/proactive_bench/env/__init__.py tests/test_smoke.py tests/test_env_trace.py README.md src/proactive_bench/cli.py
git commit -m "Phase 2: scripted tool trace, env exports, status bump"
```

---

## Self-review notes

- **Spec coverage:** §3.1 four apps + allowlist (Tasks 2–6), deep-copy isolation (Task 7), unknown-tool error type (Tasks 1/6/7), §3.4 levels + deterministic unit-tested classifier (Task 8), §3.5 command/event shape tolerated by `parse_command` (Task 6), Phase 2 exit "scripted tool trace executes" (Task 9). CLI/README status bump mirrors Phase 1's final commit.
- **Type consistency:** `ToolSpec`, `Command`, `Handler`, `ToolKind`, `Workspace.execute(Command) -> dict`, `classify_action_level(Command | Mapping | None) -> ActionLevel` used identically across tasks.
- **No new dependencies; offline tests only.**

