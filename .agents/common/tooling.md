<!-- agent-kit: common/tooling.md v0.1.3 -->
# Optional tools and plugins (agent-kit common, loaded on demand)

No model, provider, plugin, editor or indexer is required. Start from repository search, source files, runtime
evidence and official documentation. Install nothing automatically.

| Capability | Policy |
| --- | --- |
| Workflow packs (Superpowers, Ponytail, Matt Pocock's skills...) | Optional user-level plugins. Disable per project when they overlap the kit (`enabledPlugins: false` in `.claude/settings.json`). Never vendor their files. |
| Platform skills (cloud vendors, frameworks) | Enable only in projects that use the platform; prefer one official source. |
| Code index / graph tools | Optional, only when freshness and retrieval value are demonstrated; direct search is the default. |
| Review / security plugins | Optional expertise; never a substitute for the project's own checks. |
| LSPs, browser tools, MCP servers | Personal capabilities; no project depends on them. |

Reassess a tool by observable value, overlap, maintenance and permissions, not by branding. Verify actual
discovery and hook behavior in each client before claiming compatibility.
