# proactive-bench — Design Spec

**Mini benchmark for proactive engineering agents (HALO-lite)**  
Status: Approved design · Date: 2026-11-03 · Path: Architectural

---

## 1. Goal

Build a small, credible benchmark that evaluates **when a proactive AI agent should act, ask, or wait** in software-team workflows — inspired by [HALO (arXiv:2610.39086)](https://arxiv.org/abs/2610.39086), but scoped for a solo fresher AI-engineer portfolio.

**Resume artifact:** GitHub repo + results report (HTML/MD) demonstrating agent-eval harness design, multi-round tool use, LLM-as-judge methodology, and analysis of timing/intensity failure modes.

**Non-goals (v1):** Hugging Face dataset publish, human annotation study, paper writeup, multi-domain breadth, production-grade auth/rate-limit infra.

---

## 2. Success criteria

| # | Criterion |
|---|---------|
| 1 | Repo with README (badges, architecture diagram, findings), MIT/Apache license, green CI (pytest) |
| 2 | ~30 scenarios across 5 workspaces × 4 need types, validated by a schema checker |
| 3 | Multi-round agent (inspect → final) evaluated on 2–3 models via OmniRoute |
| 4 | LLM-as-judge produces acceptability, timing, intensity + rationale per checkpoint |
| 5 | `results/report.html` + `results/summary.md` with model rankings and failure-mode breakdown |
| 6 | Unit tests pass with **mocked HTTP** (no network in CI) |

---

## 3. Domain & simulated environments

**Domain:** software team workflows (Flavor 1 — dev-workflow).  
**Industry motivation:** autonomous coding/ops agents fail on *judgment* (when to re-run CI, when to escalate, when not to force-merge) more than on raw capability. This benchmark targets that calibration gap.

### 3.1 Simulated apps (4 thin, JSON/dict-backed backends)

| App id | Name | Read tools (inspect) | Write actions (final) |
|--------|------|----------------------|------------------------|
| `repo` | GitHub-like | `list_prs`, `get_pr`, `get_checks`, `list_issues`, `get_issue` | `post_comment`, `request_review`, `retrigger_ci`, `open_draft_pr`, `merge_pr`, `revert_commit` |
| `chat` | Team chat | `list_channels`, `get_thread`, `search_messages` | `send_message`, `post_standup_update`, `pin_message` |
| `pager` | On-call | `list_alerts`, `get_alert`, `get_runbook`, `get_deploy_window` | `ack_alert`, `page_oncall`, `propose_rollback`, `execute_rollback` |
| `cal` | Calendar | `list_events`, `get_event`, `get_free_slots` | `create_event`, `decline_event`, `propose_reschedule`, `block_focus_time` |

Rules:
- State is plain JSON/dict; each checkpoint gets a **fresh deep copy** (isolation, like HALO temp dirs).
- Tools are validated against an allowlist; unknown tool → agent error (bounded retry).
- No real network, no real git, no real Slack.

### 3.2 Workspaces (×5)

Each workspace = one engineer persona with coherent data in all 4 apps:

1. **backend feature dev** — PRs, flaky CI, sprint tickets  
2. **SRE on-call** — alerts, runbooks, deploy windows  
3. **tech lead** — review queue, standups, team calendar  
4. **frontend** — UI regressions, design reviews, release trains  
5. **platform/infra** — CVEs, dependency PRs, build fleet  

Scenarios are authored per workspace; need types are balanced across the set.

### 3.3 Need types (from HALO)

- **dependency** — blocked on someone/something else  
- **blockage** — something is broken/stuck  
- **risk** — bad thing likely if no action  
- **opportunity** — useful action available if timing is right  

### 3.4 Action levels (L0–L3)

| Level | Meaning | Examples |
|-------|---------|----------|
| L0 | Observe / no intervention | read-only notes, do nothing user-visible |
| L1 | Notify / inform | chat message, comment, ack, “FYI” |
| L2 | Prepare / draft | draft PR, draft rollback plan, propose reschedule |
| L3 | Execute | merge, send as self, execute rollback, decline+create |

Classifier: deterministic function of final tool + args (unit-tested), not LLM-judged.

### 3.5 Scenario JSON schema (conceptual)

```json
{
  "id": "backend-ci-freeze-01",
  "workspace": "backend",
  "need_type": "blockage",
  "persona": { "name": "...", "role": "...", "team": "..." },
  "apps": { "repo": { ...initial state... }, "chat": {...}, "pager": {...}, "cal": {...} },
  "events": [
    { "event_id": "e1", "time": "2026-11-10T09:00:00Z", "app": "repo", "function": "...", "parameter": { ... } },
    { "event_id": "e2", "time": "2026-11-10T09:15:00Z", "app": "chat", "function": "...", "parameter": { ... } }
  ],
  "notes": "optional construction-side notes (never shown to agent)"
}
```

**Checkpoints:** after applying events `[e1..ek]`, the agent is invoked once (k = 1..N). Same pattern as HALO prefixes.

---

## 4. Agent protocol

**Input:** persona + current tool allowlist + event prefix + inspection results so far.  
**Output (strict JSON):**

```json
{ "decision": "inspect" | "final", "content": [ { "app": "...", "function": "...", "parameter": {...} } ] | { ... } | null }
```

- `inspect`: one or more **read** tools; results appended to trace; loop continues (max **5** rounds).
- `final`: exactly **one** write action or `null` (do nothing → L0).
- Invalid JSON / unknown tool / write-as-inspect → retry up to **3** times → checkpoint `error`.
- Temperature: low/0 for agents where supported; reasoning effort configurable per model.

---

## 5. Judge (LLM-as-judge)

**Input:** workspace context, event prefix, full inspection trace, final action + execution result.  
**Output (strict JSON):**

```json
{
  "verdict": "acceptable" | "unacceptable",
  "action_level": "L0" | "L1" | "L2" | "L3",
  "timing": "timely" | "premature" | "belated" | "inaction",
  "intensity": "appropriate" | "insufficient" | "excessive" | null,
  "rationale": "..."
}
```

- `intensity` may be null for pure L0 inaction judgments (mirror HALO).
- Judge model ≠ primary agent model where possible (configurable in `config.yaml`).
- temperature=0, JSON-only; parse failures retried then marked judge_error.
- **Not** human ground truth — document this limitation in README (“LLM-judge proxy”).

---

## 6. Models (OmniRoute)

OpenAI-compatible endpoint (credentials via `.env`, same pattern as `D:\Code\Rag Eval`):

| Role | Default candidates (configurable) |
|------|-----------------------------------|
| Agent A | `rag` (Mistral Ministral 14B) |
| Agent B | `auto/best-fast` or `aug/haiku4.5` |
| Agent C (optional) | `auto/best-reasoning` |
| Judge | `auto/best-reasoning` or `aug/sonnet5` (must differ from at least one agent) |

Env vars: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (+ optional `JUDGE_MODEL`). Never commit `.env`; log host only.

---

## 7. Architecture & data flow

```
scenarios/*.json
      │
      ▼
  runner ──► per checkpoint: copy app state → apply event prefix
      │
      ▼
  agent loop (llm client ↔ tools) → final action | null
      │
      ▼
  judge → verdict JSON
      │
      ▼
  results/{model}/{run_id}.json  (+ fingerprints for resume)
      │
      ▼
  report → results/report.html + results/summary.md
```

**Package layout:**

```
proactive-bench/
├── src/proactive_bench/
│   ├── llm/          # OpenAI-compat client, retries, JSON parse
│   ├── env/          # apps: repo, chat, pager, cal + command parser
│   ├── agent/        # loop, prompt, action-level classifier
│   ├── judge/        # judge prompt + parser
│   ├── runner/       # orchestration, isolation, cache/resume
│   ├── report/       # aggregates, HTML/MD generation
│   └── cli.py        # validate | run | report | smoke
├── scenarios/        # workspace data + scenario JSON
├── config.yaml       # models, limits, paths
├── tests/            # unit + integration (mocked transport)
├── results/          # gitignored run outputs
├── .env.example
├── pyproject.toml
└── README.md
```

**Engineering hygiene (from Rag Eval patterns):**
- httpx client; exponential backoff on 429/5xx; honor `Retry-After`
- Input fingerprints → skip completed (model, scenario) pairs; safe resume
- Bounded parallel workers (default 4; configurable)
- Secrets never in logs or committed files

---

## 8. Report contents

- Acceptability rate by model (overall + by need_type + by workspace)
- Failure-mode breakdown among rejections: timing vs intensity (HALO-style headline stat)
- Action-level distribution (L0–L3) per model
- Average retrieval rounds / tokens per accepted vs rejected
- Top example traces (link into raw JSON)
- Clear “limitations: LLM-judge, mini scale” callout

Charts: static (matplotlib or pure HTML/CSS tables — prefer zero-heavy-deps HTML tables + one PNG if needed).

---

## 9. Testing & CI

| Layer | What | Network |
|-------|------|---------|
| Unit | tool allowlist, action-level classifier, JSON response validation, judge schema, fingerprinting | none (mock httpx) |
| Scenario | schema validator over `scenarios/**` | none |
| Integration | 1 scenario × 1 model × mock transport end-to-end; report smoke on fixture results | none |
| Manual smoke | `cli smoke` against real OmniRoute, 1×1 | yes (local only) |

CI: GitHub Actions — ruff + pytest on push/PR. Python 3.11+.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Scenario authoring burnout (~30) | Ship 10 first (Phase 3 exit), grow to 30; templates per need_type |
| Judge quality noise | Strong judge model, temp 0, rationales stored; README limitation note |
| OmniRoute flakiness / rate limits | Retries, low parallelism, resume/cache |
| Scope creep (dashboard, HF, humans) | Non-goals locked; Phase 9 is polish only |
| Model too weak for multi-round (`rag`) | Cap rounds at 5; treat errors as data, not crashes |

---

## 11. Phases (implementation order)

| Phase | Deliverable | Exit criteria |
|-------|-------------|---------------|
| **0 — Scaffold** | `pyproject.toml`, package layout, `.env.example`, `config.yaml`, ruff/pytest, CI skeleton | `pytest` collects; lint clean |
| **1 — LLM client** | `llm/` client, retries, JSON helpers, mocked tests | tests green offline |
| **2 — Simulated env** | 4 apps + command parser + isolation + action-level classifier + tests | scripted tool trace executes |
| **3 — Scenarios** | schema + 5 workspaces; ≥10 scenarios now → ~30 by end of phase | `cli validate` passes all |
| **4 — Agent loop** | inspect/final protocol, prompts, retries + mocked tests | mocked agent finishes a scenario |
| **5 — Judge** | judge prompt, parser, mocked tests | mocked judge returns valid verdicts |
| **6 — Runner** | orchestration, fingerprints, cache/resume, workers + smoke | `cli smoke` real API 1×1 works |
| **7 — Full run** | 2–3 agents × all scenarios × judge | `results/` complete, resumable |
| **8 — Report** | HTML + summary.md + stats | numbers match raw results |
| **9 — Polish** | README (badges, arch, findings), license, final CI green | repo resume-ready |

**Dependency notes:** 1–2 parallel-safe; 3 needs 2’s schema; 4 needs 1+2; 5 needs 1; 6 needs 4+5; 7–8 need 6; 9 last.

---

## 12. Explicit design decisions (locked)

1. **Approach A — HALO-lite**, not harness-first or dashboard-first  
2. **Domain:** dev-workflow (repo/chat/pager/cal), not personal-assistant  
3. **Judge:** LLM-only (no human gold set in v1)  
4. **Scope:** mini — 5 workspaces, ~30 scenarios, 2–3 agents + 1 judge  
5. **Agency:** multi-round retrieval, max 5 inspect rounds  
6. **Deliverables:** GitHub repo (docs+tests) + results report only  
7. **Endpoint:** OmniRoute via env vars (pattern copied from Rag Eval `llm_client`)  
8. **Spec location:** `D:\Code\Agentic Eval\SPEC.md` (this file)

---

## 13. Open items for implementation plan

- Exact workspace/fixture JSON shapes (decide in Phase 2–3)  
- Final default model IDs after a one-shot OmniRoute capability probe  
- Chart approach: HTML-only vs matplotlib PNG  
- License choice (MIT vs Apache-2.0)  

---

*Next step after spec approval: writing-plans → step-by-step implementation plan → Phase 0.*
