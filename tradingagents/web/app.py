"""Local browser UI for TradingAgents.

The web app is intentionally thin: FastAPI serves a static single-page UI and a
small JSON API. Analysis runs execute the existing LangGraph engine on a worker
thread and expose progress snapshots to the browser.
"""

from __future__ import annotations

import copy
import importlib
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from cli.stats_handler import StatsCallbackHandler
from tradingagents import default_config as default_config_module
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.graph.trading_graph import TradingAgentsGraph

STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_CONFIG = default_config_module.DEFAULT_CONFIG

EFFORT_BY_DEPTH = {
    1: "low",
    3: "high",
    5: "xhigh",
}

DEPTH_LABELS = {
    1: "Shallow",
    3: "Medium",
    5: "Deep",
}

PROVIDER_DEFAULTS = {
    "codex_cli": {
        "quick_model": "gpt-5.5",
        "deep_model": "gpt-5.5",
    },
    "claude_cli": {
        "quick_model": "claude-sonnet-4-6",
        "deep_model": "claude-opus-4-8",
    },
}

ANALYST_LABELS = {
    "market": "Market Analyst",
    "social": "Social Media Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

FIXED_AGENT_LABELS = [
    "Bull/Bear Advocates",
    "Research Evaluator",
    "Trader",
    "Risk Analysts",
    "Portfolio Manager",
]

REPORT_TO_AGENT = {
    "market_report": "Market Analyst",
    "sentiment_report": "Social Media Analyst",
    "news_report": "News Analyst",
    "fundamentals_report": "Fundamentals Analyst",
    "investment_plan": "Research Evaluator",
    "trader_investment_plan": "Trader",
    "final_trade_decision": "Portfolio Manager",
}

REPORT_TITLES = {
    "market_report": "Market Analyst Report",
    "sentiment_report": "Social Media Report",
    "news_report": "News Analyst Report",
    "fundamentals_report": "Fundamentals Report",
    "investment_plan": "Research Evaluation",
    "trader_investment_plan": "Trading Plan",
    "final_trade_decision": "Final Trade Decision",
}


class RunCancelled(RuntimeError):
    """Raised inside the worker when the user cancels a browser run."""


class CancelCallbackHandler(BaseCallbackHandler):
    raise_error = True

    def __init__(self, handle: RunHandle) -> None:
        super().__init__()
        self.handle = handle

    def _check_cancelled(self) -> None:
        if self.handle.cancel_requested:
            raise RunCancelled("Analysis cancelled")

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        self._check_cancelled()

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._check_cancelled()

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        self._check_cancelled()

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._check_cancelled()


class RunRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    analysis_date: str
    analysts: list[str] = Field(default_factory=lambda: ["market", "social", "news", "fundamentals"])
    research_depth: int = 5
    llm_provider: str | None = "codex_cli"
    quick_model: str | None = None
    deep_model: str | None = None
    asset_type: str = "stock"
    output_language: str = "English"


class RunHandle:
    def __init__(self, request: RunRequest):
        self.run_id = uuid.uuid4().hex[:12]
        self.request = request
        self.status = "queued"
        self.started_at = time.time()
        self.completed_at: float | None = None
        self.error: str | None = None
        self.stats: dict[str, Any] = {
            "llm_calls": 0,
            "tool_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        }
        self.agents = self._initial_agents(request.analysts)
        self.reports: dict[str, str] = {}
        self.current_report_key: str | None = None
        self.final_decision = ""
        self.signal: str | None = None
        self.report_path: str | None = None
        self.activity: list[dict[str, Any]] = []
        self.config = _build_run_config(request)
        self.cancel_requested = False
        self._lock = threading.Lock()

    @staticmethod
    def _initial_agents(analysts: list[str]) -> list[dict[str, str]]:
        out = [{"name": ANALYST_LABELS[key], "status": "pending"} for key in analysts if key in ANALYST_LABELS]
        out.extend({"name": name, "status": "pending"} for name in FIXED_AGENT_LABELS)
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = int((self.completed_at or time.time()) - self.started_at)
            return {
                "run_id": self.run_id,
                "status": self.status,
                "elapsed_seconds": elapsed,
                "error": self.error,
                "request": self.request.model_dump(),
                "config": {
                    "llm_provider": self.config["llm_provider"],
                    "quick_model": self.config["quick_think_llm"],
                    "deep_model": self.config["deep_think_llm"],
                    "research_depth": self.config["max_debate_rounds"],
                    "research_depth_label": DEPTH_LABELS.get(
                        self.config["max_debate_rounds"], "Custom"
                    ),
                    "local_cli_effort": self.config.get("local_cli_effort"),
                },
                "agents": list(self.agents),
                "reports": dict(self.reports),
                "report_titles": REPORT_TITLES,
                "current_report_key": self.current_report_key,
                "final_decision": self.final_decision,
                "signal": self.signal,
                "report_path": self.report_path,
                "activity": list(self.activity[-80:]),
                "stats": dict(self.stats),
            }

    def mark_running(self) -> None:
        with self._lock:
            if self.cancel_requested:
                return
            self.status = "running"
            self._add_activity_locked("Run started", "Preparing analysis engine")

    def mark_error(self, exc: BaseException) -> None:
        with self._lock:
            if self.cancel_requested:
                self._mark_cancelled_locked()
                return
            self.status = "error"
            self.error = str(exc)
            self.completed_at = time.time()
            self._add_activity_locked("Run failed", self.error)

    def mark_done(self, final_state: dict[str, Any], signal: str, report_path: Path) -> None:
        with self._lock:
            if self.cancel_requested:
                self._mark_cancelled_locked()
                return
            self.status = "completed"
            self.signal = signal
            self.final_decision = str(final_state.get("final_trade_decision") or "")
            if self.final_decision:
                self.reports["final_trade_decision"] = self.final_decision
                self.current_report_key = "final_trade_decision"
            self.report_path = str(report_path)
            self.completed_at = time.time()
            self._set_agent_locked("Portfolio Manager", "completed")
            self._add_activity_locked("Portfolio Manager", "Final decision completed")

    def cancel(self) -> bool:
        with self._lock:
            if self.status in {"completed", "error", "cancelled"}:
                return False
            self.cancel_requested = True
            self._mark_cancelled_locked()
            return True

    def update_stats(self, stats: dict[str, Any]) -> None:
        with self._lock:
            self.stats = dict(stats)

    def update_from_chunk(self, chunk: dict[str, Any], stats: dict[str, Any]) -> None:
        with self._lock:
            if self.cancel_requested:
                raise RunCancelled("Analysis cancelled")
            self.stats = dict(stats)
            self._mark_active_agent_locked(chunk)
            for key, agent in REPORT_TO_AGENT.items():
                value = chunk.get(key)
                if not value:
                    continue
                self.reports[key] = str(value)
                self.current_report_key = key
                self._set_agent_locked(agent, "completed")
                self._add_activity_locked(agent, f"{REPORT_TITLES[key]} ready")

            debate = chunk.get("investment_debate_state") or {}
            if debate.get("bull_history") or debate.get("bear_history"):
                self._set_agent_locked("Bull/Bear Advocates", "completed")
            if debate.get("judge_decision"):
                self.reports["investment_plan"] = str(debate["judge_decision"])
                self.current_report_key = "investment_plan"
                self._set_agent_locked("Research Evaluator", "completed")
                self._add_activity_locked("Research Evaluator", "Research decision ready")

            risk = chunk.get("risk_debate_state") or {}
            if risk.get("aggressive_history") or risk.get("conservative_history") or risk.get("neutral_history"):
                self._set_agent_locked("Risk Analysts", "completed")
            if risk.get("judge_decision"):
                self.reports["final_trade_decision"] = str(risk["judge_decision"])
                self.current_report_key = "final_trade_decision"

    def _mark_active_agent_locked(self, chunk: dict[str, Any]) -> None:
        order = [agent["name"] for agent in self.agents]
        for name in order:
            status = self._agent_status_locked(name)
            if status == "pending":
                self._set_agent_locked(name, "running")
                return

    def _agent_status_locked(self, name: str) -> str | None:
        for agent in self.agents:
            if agent["name"] == name:
                return agent["status"]
        return None

    def _set_agent_locked(self, name: str, status: str) -> None:
        for agent in self.agents:
            if agent["name"] == name:
                agent["status"] = status
                return

    def _add_activity_locked(self, agent: str, message: str) -> None:
        self.activity.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "agent": agent,
                "message": message,
                "stats": dict(self.stats),
            }
        )

    def _mark_cancelled_locked(self) -> None:
        if self.status != "cancelled":
            self.status = "cancelled"
            self.error = None
            self.completed_at = time.time()
            self._add_activity_locked("Run cancelled", "Analysis stopped by user")


app = FastAPI(title="Local Trading Desk Web UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_runs: dict[str, RunHandle] = {}
_runs_lock = threading.Lock()


def _previous_day() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def _build_run_config(request: RunRequest) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["llm_provider"] = (request.llm_provider or config["llm_provider"]).lower()
    config["quick_think_llm"] = request.quick_model or config["quick_think_llm"]
    config["deep_think_llm"] = request.deep_model or config["deep_think_llm"]
    config["max_debate_rounds"] = int(request.research_depth)
    config["max_risk_discuss_rounds"] = int(request.research_depth)
    config["local_cli_effort"] = EFFORT_BY_DEPTH.get(int(request.research_depth), "low")
    config["output_language"] = request.output_language or "English"
    config["checkpoint_enabled"] = False
    # Keep the browser demo on keyless/free data by default. Users can still
    # change data vendor config in code/env for deeper local runs.
    data_vendors = dict(config.get("data_vendors") or {})
    data_vendors.setdefault("core_stock_apis", "yfinance")
    data_vendors.setdefault("technical_indicators", "yfinance")
    data_vendors.setdefault("fundamental_data", "yfinance")
    data_vendors.setdefault("news_data", "yfinance")
    config["data_vendors"] = data_vendors
    return config


def _child_processes(root_pid: int) -> dict[int, str]:
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    children_by_parent: dict[int, list[int]] = {}
    commands: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        commands[pid] = parts[2] if len(parts) == 3 else ""
        children_by_parent.setdefault(ppid, []).append(pid)

    descendants: dict[int, str] = {}
    stack = list(children_by_parent.get(root_pid, []))
    while stack:
        pid = stack.pop()
        descendants[pid] = commands.get(pid, "")
        stack.extend(children_by_parent.get(pid, []))
    return descendants


def _terminate_local_cli_children() -> int:
    targets = []
    for pid, command in _child_processes(os.getpid()).items():
        normalized = command.lower()
        if "codex" not in normalized and "claude" not in normalized:
            continue
        targets.append(pid)

    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    time.sleep(0.5)

    killed = 0
    for pid in targets:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            killed += 1
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except ProcessLookupError:
            killed += 1
    return killed


def _run_analysis(handle: RunHandle) -> None:
    try:
        handle.mark_running()
        set_config(handle.config)
        stats = StatsCallbackHandler()
        cancel = CancelCallbackHandler(handle)
        graph = TradingAgentsGraph(
            selected_analysts=handle.request.analysts,
            config=handle.config,
            debug=False,
            callbacks=[stats, cancel],
        )
        ticker = handle.request.ticker.upper().strip()
        trade_date = handle.request.analysis_date
        graph.ticker = ticker
        graph._resolve_pending_entries(ticker)
        past_context = graph.memory_log.get_past_context(ticker)
        instrument_context = graph.resolve_instrument_context(ticker, handle.request.asset_type)
        initial_state = graph.propagator.create_initial_state(
            ticker,
            trade_date,
            asset_type=handle.request.asset_type,
            past_context=past_context,
            instrument_context=instrument_context,
        )
        args = graph.propagator.get_graph_args()
        final_state: dict[str, Any] = {}
        for chunk in graph.graph.stream(initial_state, **args):
            if handle.cancel_requested:
                raise RunCancelled("Analysis cancelled")
            final_state.update(chunk)
            handle.update_from_chunk(chunk, stats.get_stats())

        if handle.cancel_requested:
            raise RunCancelled("Analysis cancelled")

        graph.curr_state = final_state
        graph._log_state(trade_date, final_state)
        graph.memory_log.store_decision(
            ticker=ticker,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )
        signal = graph.process_signal(final_state["final_trade_decision"])
        report_path = graph.save_reports(
            final_state,
            ticker,
            Path(handle.config["results_dir"])
            / safe_ticker_component(ticker)
            / trade_date
            / "web_reports",
        )
        handle.mark_done(final_state, signal, report_path)
    except RunCancelled:
        handle.cancel()
    except Exception as exc:  # noqa: BLE001 - the worker must report failures to the UI.
        handle.mark_error(exc)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/r/{run_id}")
async def run_page(run_id: str) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "runs": len(_runs)}


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "default_date": _previous_day(),
        "llm_provider": DEFAULT_CONFIG["llm_provider"],
        "quick_model": DEFAULT_CONFIG["quick_think_llm"],
        "deep_model": DEFAULT_CONFIG["deep_think_llm"],
        "analysts": ANALYST_LABELS,
        "providers": ["codex_cli", "claude_cli", "openai", "anthropic", "google", "ollama"],
        "provider_defaults": PROVIDER_DEFAULTS,
        "depth_efforts": EFFORT_BY_DEPTH,
    }


@app.post("/api/runs")
async def create_run(request: RunRequest) -> dict[str, str]:
    analysts = [key for key in request.analysts if key in ANALYST_LABELS]
    if not analysts:
        raise HTTPException(status_code=400, detail="Select at least one analyst")
    request.analysts = analysts
    request.ticker = request.ticker.upper().strip()
    handle = RunHandle(request)
    with _runs_lock:
        _runs[handle.run_id] = handle
    thread = threading.Thread(target=_run_analysis, args=(handle,), daemon=True)
    thread.start()
    return {"run_id": handle.run_id}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    handle = _runs.get(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return handle.snapshot()


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    handle = _runs.get(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail="Run not found")
    cancelled = handle.cancel()
    killed = _terminate_local_cli_children()
    payload = handle.snapshot()
    payload["cancelled"] = cancelled
    payload["terminated_processes"] = killed
    return payload


def main() -> None:
    import uvicorn

    load_dotenv()
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = importlib.reload(default_config_module).DEFAULT_CONFIG
    port = int(os.environ.get("TRADINGAGENTS_WEB_PORT", "8501"))
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
