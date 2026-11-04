# proactive-bench

> **When should an AI agent act, ask, or wait?**  
> A mini benchmark for **proactive engineering agents** — inspired by [HALO (arXiv:2610.39086)](https://arxiv.org/abs/2610.39086).

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: Phase 1](https://img.shields.io/badge/status-phase%201-yellow)
![tests](https://img.shields.io/badge/tests-pytest-informational)

---

## Why this exists

Coding and ops agents are getting good at *doing* tasks — and bad at *judging when to intervene*. Force-merging red CI, pinging teammates at 2am, or staying silent through a freeze are **calibration** failures, not capability failures.

`proactive-bench` evaluates that judgment in simulated software-team workflows:

- **4 apps** a developer actually lives in: `repo` (PRs/CI) · `chat` · `pager` · `cal`
- **5 workspaces** (backend, SRE, lead, frontend, platform) × **~30 scenarios**
- **4 need types:** dependency · blockage · risk · opportunity
- **Multi-round agents:** inspect (read tools) → final (act or do nothing), action levels **L0–L3**
- **LLM-as-judge:** acceptability, timing (`timely/premature/belated/inaction`), intensity (`appropriate/insufficient/excessive`)
- **Results report:** model rankings + failure-mode breakdown (HALO-style headline stats)

> Not a personal-assistant toy — scenarios are real eng life: flaky CI before a freeze, CVE PRs, stale reviews, deploy vs retro conflicts, on-call handoffs.

---

## Architecture

```
scenarios/*.json
      │
      ▼
  runner ──► per checkpoint: copy app state → apply event prefix
      │
      ▼
  agent loop (LLM ↔ tools) → final action | null
      │
      ▼
  judge → { verdict, timing, intensity, rationale }
      │
      ▼
  results/{model}/{run_id}.json   (fingerprints → resume-safe)
      │
      ▼
  report → results/report.html + results/summary.md
```

```
src/proactive_bench/
├── llm/      # OpenAI-compat client (OmniRoute-friendly), retries, JSON
├── env/      # repo · chat · pager · cal + command parser + isolation
├── agent/    # inspect→final loop, prompts, L0–L3 classifier
├── judge/    # structured verdicts
├── runner/   # orchestration, cache/resume, workers
├── report/   # aggregates + HTML/MD
└── cli.py    # validate | run | report | smoke
```

Full design: [`SPEC.md`](./SPEC.md).

---

## Quick start

```bash
# 1. Environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Secrets (OpenAI-compatible endpoint)
cp .env.example .env             # then edit LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

# 3. Checks
ruff check src tests
pytest
```

CLI (subcommands light up as phases land):

```bash
proactive-bench --version
proactive-bench validate    # scenario schema (Phase 3)
proactive-bench smoke       # 1×1 real-API smoke (Phase 6)
proactive-bench run         # full matrix (Phase 7)
proactive-bench report      # HTML + summary (Phase 8)
```

---

## Roadmap

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 0 | Scaffold · lint · tests · CI · README | ✅ |
| 1 | LLM client + mocked tests | ✅ **current** |
| 2 | Simulated apps + parser + L0–L3 | ⬜ |
| 3 | Scenarios (5 workspaces → ~30) | ⬜ |
| 4 | Multi-round agent loop | ⬜ |
| 5 | LLM judge | ⬜ |
| 6 | Runner + smoke | ⬜ |
| 7 | Full model × scenario run | ⬜ |
| 8 | HTML/MD report | ⬜ |
| 9 | Polish + final CI green | ⬜ |

Details and exit criteria: [`SPEC.md` §11](./SPEC.md).

---

## Design principles

1. **Offline-first tests** — mocked HTTP in CI; real API only for local smoke/full runs  
2. **Secrets never committed** — `.env` gitignored; log hosts only  
3. **Isolated checkpoints** — each event prefix starts from a fresh app-state copy  
4. **Honest limitations** — LLM-judge proxy, mini scale; stated in every report  
5. **Resume-safe** — input fingerprints skip completed (model, scenario) pairs  

---

## Credits

- Benchmark design inspired by **HALO: Human-Grounded Granular Evaluation of Proactive Agents** ([paper](https://arxiv.org/abs/2610.39086), [code](https://github.com/CedricPei/HALO)) — this project is an independent, smaller re-implementation for learning/portfolio purposes, not a fork.
- LLM client patterns adapted from the author's `rag-eval` harness (OpenAI-compatible env config).

## License

[MIT](./LICENSE)
