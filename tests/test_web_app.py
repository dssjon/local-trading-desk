from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import tradingagents.web.app as web_app


@pytest.fixture
def client():
    web_app._runs.clear()
    return TestClient(web_app.app)


def test_web_index_and_config_load(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "Target Equity" in index.text
    assert "Start Analysis" in index.text
    assert "Research Depth" in index.text
    assert "Tauric Research" not in index.text

    config = client.get("/api/config")
    assert config.status_code == 200
    payload = config.json()
    assert payload["llm_provider"]
    assert "market" in payload["analysts"]
    assert payload["provider_defaults"]["codex_cli"]["quick_model"] == "gpt-5.5"
    assert payload["depth_efforts"] == {"1": "low", "3": "high", "5": "xhigh"}


def test_create_run_and_poll_with_stubbed_worker(client, monkeypatch):
    def fake_run(handle):
        handle.mark_running()
        handle.mark_done(
            {"final_trade_decision": "FINAL TRANSACTION PROPOSAL: **BUY**"},
            "BUY",
            Path("/tmp/report.md"),
        )

    monkeypatch.setattr(web_app, "_run_analysis", fake_run)

    response = client.post(
        "/api/runs",
        json={
            "ticker": "AAPL",
            "analysis_date": "2026-06-25",
            "analysts": ["market"],
            "research_depth": 1,
            "llm_provider": "codex_cli",
            "quick_model": "gpt-5.5",
            "deep_model": "gpt-5.5",
        },
    )
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    payload = None
    for _ in range(20):
        poll = client.get(f"/api/runs/{run_id}")
        assert poll.status_code == 200
        payload = poll.json()
        if payload["status"] == "completed":
            break
        time.sleep(0.05)

    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["request"]["ticker"] == "AAPL"
    assert payload["signal"] == "BUY"
    assert "final_trade_decision" in payload["reports"]
    assert payload["config"]["local_cli_effort"] == "low"


@pytest.mark.parametrize(
    "depth,expected",
    [
        (1, "low"),
        (3, "high"),
        (5, "xhigh"),
    ],
)
def test_web_depth_sets_distinct_local_cli_effort(depth, expected):
    request = web_app.RunRequest(
        ticker="MSFT",
        analysis_date="2026-06-25",
        research_depth=depth,
        llm_provider="claude_cli",
        quick_model="claude-sonnet-4-6",
        deep_model="claude-opus-4-8",
    )
    config = web_app._build_run_config(request)

    assert config["max_debate_rounds"] == depth
    assert config["max_risk_discuss_rounds"] == depth
    assert config["local_cli_effort"] == expected


def test_cancel_run_endpoint_marks_run_cancelled(client, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fake_run(handle):
        handle.mark_running()
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(web_app, "_run_analysis", fake_run)
    monkeypatch.setattr(web_app, "_terminate_local_cli_children", lambda: 0)

    response = client.post(
        "/api/runs",
        json={
            "ticker": "AAPL",
            "analysis_date": "2026-06-25",
            "analysts": ["market"],
            "research_depth": 5,
            "llm_provider": "codex_cli",
            "quick_model": "gpt-5.5",
            "deep_model": "gpt-5.5",
        },
    )
    assert response.status_code == 200
    assert started.wait(timeout=2)
    run_id = response.json()["run_id"]

    cancel = client.post(f"/api/runs/{run_id}/cancel")
    release.set()

    assert cancel.status_code == 200
    payload = cancel.json()
    assert payload["status"] == "cancelled"
    assert payload["config"]["local_cli_effort"] == "xhigh"
