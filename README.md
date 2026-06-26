# Local Trading Desk

Local Trading Desk is a refactored version of the open source
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
framework. This fork keeps the multi-agent financial research engine, adds
subscription-backed `codex` and `claude` CLI providers, and ships a local browser
interface inspired by the trading-agents.ai web app.

This project is for research and education only. It is not financial,
investment, or trading advice.

## What This Fork Adds

- Local web UI at `http://127.0.0.1:8501/`.
- Codex CLI and Claude CLI LLM providers that use your existing CLI login rather
  than hosted LLM API keys.
- Provider picker in the browser for `Codex CLI` or `Claude CLI`.
- Research depth controls that map to local CLI effort:
  - `Shallow` -> `low`
  - `Medium` -> `high`
  - `Deep` -> `xhigh`
- Stop button for cancelling accidental browser runs and terminating local CLI
  child processes.
- Safer `.env` defaults and setup docs for local use.

## License And Attribution

This repository is a modified derivative of TradingAgents, licensed under the
Apache License 2.0. The original Apache 2.0 license is preserved in
[LICENSE](LICENSE). See [NOTICE](NOTICE) and [MODIFICATIONS.md](MODIFICATIONS.md)
for attribution and a summary of significant changes.

This fork is not affiliated with Tauric Research or trading-agents.ai. The local
browser UI is an independent implementation inspired by the public site design.

## Quick Start With Codex CLI

Prerequisites:

- Python 3.10 or newer.
- A working `codex` CLI installation.
- A logged-in Codex CLI session on the machine running this app.

Install and run:

```bash
git clone https://github.com/dssjon/local-trading-desk.git
cd local-trading-desk

python -m venv .venv
source .venv/bin/activate
pip install -e ".[web]"

cp .env.example .env
```

Edit `.env` and set the local CLI provider:

```bash
TRADINGAGENTS_LLM_PROVIDER=codex_cli
TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.5
TRADINGAGENTS_WEB_PORT=8501
```

Start the browser UI:

```bash
tradingagents-web
```

Open:

```text
http://127.0.0.1:8501/
```

The first screen is the working dashboard. Set the analysis date, analyst team,
LLM provider, and research depth before entering a ticker and pressing
`Start Analysis`.

## Claude CLI Option

If you prefer Claude Code / Claude CLI, log in with `claude` first, then set:

```bash
TRADINGAGENTS_LLM_PROVIDER=claude_cli
TRADINGAGENTS_DEEP_THINK_LLM=claude-opus-4-8
TRADINGAGENTS_QUICK_THINK_LLM=claude-sonnet-4-6
TRADINGAGENTS_WEB_PORT=8501
```

The browser provider toggle can switch between Codex and Claude. The model
fields are editable before each run.

## Optional Market Data Keys

The app can run with free/keyless Yahoo Finance-backed data for many workflows.
For richer data, copy `.env.example` to `.env` and fill in optional keys such as:

```bash
ALPHA_VANTAGE_API_KEY=
FRED_API_KEY=
```

Do not commit `.env`; it is intentionally ignored.

## CLI Usage

The original interactive terminal flow is still available:

```bash
tradingagents
```

The browser UI is usually the easier entrypoint:

```bash
tradingagents-web
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev,web]"
```

Run checks:

```bash
ruff check .
python -m pytest
node --check tradingagents/web/static/app.js
```

## Notes For Public Forks

This repo intentionally removes upstream company branding from the local web UI
surface while preserving license attribution in the repository. If you publish
your own derivative, keep the Apache 2.0 license, preserve attribution, and make
your modifications clear.
