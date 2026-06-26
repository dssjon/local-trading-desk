"""Headless TradingAgents runner for local automation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta
from typing import Any

from dotenv import load_dotenv

from tradingagents.web.app import RunHandle, RunRequest, _run_analysis

DEFAULT_ANALYSTS = ["market", "social", "news", "fundamentals"]


def _default_date() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def _parse_analysts(value: str) -> list[str]:
    analysts = [item.strip() for item in value.split(",") if item.strip()]
    return analysts or DEFAULT_ANALYSTS


def _build_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    reports = snapshot.get("reports") or {}
    ordered_report_keys = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    ]

    return {
        "success": snapshot.get("status") == "completed",
        "run_id": snapshot.get("run_id"),
        "status": snapshot.get("status"),
        "elapsed_seconds": snapshot.get("elapsed_seconds"),
        "error": snapshot.get("error"),
        "request": snapshot.get("request"),
        "config": snapshot.get("config"),
        "signal": snapshot.get("signal"),
        "final_decision": snapshot.get("final_decision"),
        "report_path": snapshot.get("report_path"),
        "stats": snapshot.get("stats"),
        "reports": {
            key: reports[key]
            for key in ordered_report_keys
            if reports.get(key)
        },
    }


def run_headless(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()

    ticker = args.ticker.upper().strip()
    request = RunRequest(
        ticker=ticker,
        analysis_date=args.analysis_date,
        analysts=args.analysts,
        research_depth=args.depth,
        llm_provider=args.provider,
        quick_model=args.quick_model,
        deep_model=args.deep_model,
        asset_type=args.asset_type or ("crypto" if "-" in ticker else "stock"),
        output_language=args.output_language,
    )

    handle = RunHandle(request)
    _run_analysis(handle)
    return _build_payload(handle.snapshot())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a non-interactive TradingAgents analysis and print the result."
    )
    parser.add_argument("ticker", help="Ticker or symbol to analyze.")
    parser.add_argument(
        "--date",
        dest="analysis_date",
        default=_default_date(),
        help="Analysis date in YYYY-MM-DD format. Defaults to yesterday.",
    )
    parser.add_argument(
        "--analysts",
        type=_parse_analysts,
        default=DEFAULT_ANALYSTS,
        help="Comma-separated analysts. Defaults to market,social,news,fundamentals.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=[1, 3, 5],
        default=5,
        help="Research depth. 5 maps to xhigh local CLI effort.",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("TRADINGAGENTS_LLM_PROVIDER", "codex_cli"),
        help="LLM provider. Defaults to codex_cli.",
    )
    parser.add_argument(
        "--quick-model",
        default=os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM", "gpt-5.5"),
        help="Quick-thinking model.",
    )
    parser.add_argument(
        "--deep-model",
        default=os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM", "gpt-5.5"),
        help="Deep-thinking model.",
    )
    parser.add_argument("--asset-type", default=None, help="stock or crypto. Defaults from symbol.")
    parser.add_argument("--output-language", default="English")
    parser.add_argument("--json", action="store_true", help="Print JSON. Currently the default.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        payload = run_headless(args)
    except Exception as exc:  # noqa: BLE001 - automation needs machine-readable failure.
        print(json.dumps({"success": False, "status": "error", "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc

    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload.get("success") else 2)


if __name__ == "__main__":
    main()
