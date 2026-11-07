# Strategy re-audit v5 — Index

Campaign of 2026-11-07: 17 units × 2 blind agents (advocate, devil's advocate), DeepSeek v4.1 Flash at max reasoning
effort, run headlessly through Command Code. Synthesis and verification by Claude.

- Synthesis and redesign proposal: [REDESIGN.md](REDESIGN.md)
- Raw reports (as delivered by the agents): [reports/](reports/)
- All findings in one machine-readable file: [findings.jsonl](findings.jsonl) (728 findings)
- Protocol: the `paired-audit` and `external-subagents` skills of agent-kit.

Scores are each agent's own judgement of the unit's soundness (0-10) from its role; they are opinions, not measures.
Critic totals: 78 critical, 181 major, 77 minor.

| Unit | Topic | Advocate score | Critic score | Critic findings (critical / major / minor) | Reports |
|---|---|---|---|---|---|
| U01 | Vision, premises, nature of risk | 5.0 | 3.0 | 5 / 10 / 3 | [advocate](reports/U01_advocate.md) · [critic](reports/U01_critic.md) |
| U02 | Prices vs fundamentals | 7.0 | 3.0 | 3 / 5 / 4 | [advocate](reports/U02_advocate.md) · [critic](reports/U02_critic.md) |
| U03 | Data sourcing, universe, PIT, cleaning | 5.0 | 2.0 | 5 / 9 / 4 | [advocate](reports/U03_advocate.md) · [critic](reports/U03_critic.md) |
| U04 | Feature engineering and windowing | 7.0 | 3.0 | 2 / 13 / 3 | [advocate](reports/U04_advocate.md) · [critic](reports/U04_critic.md) |
| U05 | VAE choice and architecture | 7.0 | 2.0 | 4 / 10 / 4 | [advocate](reports/U05_advocate.md) · [critic](reports/U05_critic.md) |
| U06 | Loss function | 7.0 | 3.0 | 1 / 14 / 4 | [advocate](reports/U06_advocate.md) · [critic](reports/U06_critic.md) |
| U07 | Training procedure | 6.0 | 3.0 | 3 / 9 / 7 | [advocate](reports/U07_advocate.md) · [critic](reports/U07_critic.md) |
| U08 | Inference, exposures B, active units | 6.0 | 3.0 | 5 / 8 / 3 | [advocate](reports/U08_advocate.md) · [critic](reports/U08_critic.md) |
| U09 | Factor risk model | 7.0 | 3.0 | 4 / 8 / 4 | [advocate](reports/U09_advocate.md) · [critic](reports/U09_critic.md) |
| U10 | Diversification objective (entropy) | 6.0 | 3.0 | 6 / 13 / 5 | [advocate](reports/U10_advocate.md) · [critic](reports/U10_critic.md) |
| U11 | Optimization machinery | 5.0 | 2.0 | 7 / 16 / 4 | [advocate](reports/U11_advocate.md) · [critic](reports/U11_critic.md) |
| U12 | Validation protocol | 4.0 | 2.0 | 4 / 16 / 5 | [advocate](reports/U12_advocate.md) · [critic](reports/U12_critic.md) |
| U13 | Benchmarks and fairness | 6.0 | 2.0 | 5 / 9 / 8 | [advocate](reports/U13_advocate.md) · [critic](reports/U13_critic.md) |
| U14 | Metrics, tests, decision logic | 4.0 | 2.0 | 9 / 12 / 9 | [advocate](reports/U14_advocate.md) · [critic](reports/U14_critic.md) |
| U15 | Parameters and roadmap | 6.0 | 2.0 | 6 / 10 / 5 | [advocate](reports/U15_advocate.md) · [critic](reports/U15_critic.md) |
| U16 | Post-mortem honesty | 5.0 | 3.0 | 5 / 13 / 2 | [advocate](reports/U16_advocate.md) · [critic](reports/U16_critic.md) |
| U17 | Spec coherence DVT/ISD/code | 7.0 | 3.0 | 4 / 6 / 3 | [advocate](reports/U17_advocate.md) · [critic](reports/U17_critic.md) |
