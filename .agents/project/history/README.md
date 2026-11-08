# History

Durable results of finished or running work: audits, validated plans, measurements, decision logs, migration
ledgers. History records what was true when it was written; it is never proof of the current code.

New results go to `history/<YYYY-MM-DD>-<topic>/` (report.md, plan.md, small evidence). Large raw outputs stay out
of git: keep a summary and the numbers here. Drafts belong to the tool's scratchpad, not here.

## Index

| Folder | Content |
|---|---|
| [2026-02-to-11-changelog/](2026-02-to-11-changelog/changelog.md) | Former `.claude/rules/project/changelog.md`: development log from February to November 2026 (V1/V2 post-mortems, fixes, workspace restore) |
| [2026-11-07-agent-config-migration/](2026-11-07-agent-config-migration/ledger.md) | Migration of `.claude/` to `AGENTS.md` + `.agents/` (ledger and checks) |
| `docs/audit_v5/` (in docs) | Strategy re-audit v5 from code and results (2026-11-07) |
| `docs/audit_theory/` (in docs) | Theory-only paired re-audit (2026-11-07); INDEX.md has verdicts and stream-cut annex |
| `docs/REDESIGN.md` (in docs) | Consolidated redesign of record, built from both audits |

Research content about the strategy (audits, redesigns, post-mortems, literature reviews, strategic paths) lives in
`docs/` next to the specification; this folder keeps agent and tooling records only (see project instructions).
