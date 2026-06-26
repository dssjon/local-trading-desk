# Modifications

This repository is a modified derivative of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).

Significant changes in this fork:

- Added local `codex_cli` and `claude_cli` LLM providers.
- Added environment and CLI configuration for subscription-backed local CLI
  model execution.
- Added a FastAPI-powered local browser UI under `tradingagents/web`.
- Added a workspace-first dashboard inspired by trading-agents.ai, without
  upstream company branding in the browser surface.
- Added browser controls for analyst team, provider, models, research depth,
  and run cancellation.
- Added research-depth to local CLI effort mapping:
  - `Shallow` -> `low`
  - `Medium` -> `high`
  - `Deep` -> `xhigh`
- Added tests for local CLI providers, browser API behavior, effort mapping,
  and environment overrides.
- Added setup documentation for local Codex CLI and Claude CLI use.
- Added generated codebase, architecture, and complexity notes under `docs/`.

The original Apache 2.0 license is preserved in `LICENSE`.
