# scenarios/

Self-contained scenario JSON files (SPEC §3.5). Validated by
`proactive-bench validate` (schema, allowlisted events, unique ids).

Layout:

```
scenarios/
├── README.md
└── <scenario-id>.json    # id must match filename stem
```

Workspaces: backend · sre · lead · frontend · platform
Need types: dependency · blockage · risk · opportunity

Phase 3 ships ≥10 scenarios; grow toward ~30 with the same schema.
