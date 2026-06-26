from __future__ import annotations

from pydantic import BaseModel

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.cli_client import (
    LocalCliChatModel,
    _build_claude_args,
    _build_codex_args,
    _clean_env_for_provider,
    _loads_json_object,
    _unwrap_structured_payload,
)
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.validators import validate_model


def test_factory_routes_local_cli_providers(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CODEX_BIN", "codex-test")
    monkeypatch.setenv("TRADINGAGENTS_CLAUDE_BIN", "claude-test")

    codex = create_llm_client("codex_cli", "gpt-5.5").get_llm()
    claude = create_llm_client("claude_cli", "claude-opus-4-8").get_llm()

    assert type(codex).__name__ == "LocalCliChatModel"
    assert codex.provider == "codex_cli"
    assert codex.command == "codex-test"
    assert codex.effort == "xhigh"
    assert codex.service_tier == "fast"

    assert type(claude).__name__ == "LocalCliChatModel"
    assert claude.provider == "claude_cli"
    assert claude.command == "claude-test"
    assert claude.effort == "xhigh"
    assert claude.service_tier is None


def test_local_cli_providers_are_keyless_and_accept_custom_models():
    assert get_api_key_env("codex_cli") is None
    assert get_api_key_env("claude_cli") is None
    assert validate_model("codex_cli", "any-codex-model") is True
    assert validate_model("claude_cli", "any-claude-model") is True


def test_clean_env_removes_api_keys_but_preserves_cli_auth_context():
    base_env = {
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "anthropic-test",
        "ANTHROPIC_MODEL": "claude-test",
        "CODEX_HOME": "/tmp/codex-home",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
        "PATH": "/bin",
    }

    codex_env = _clean_env_for_provider("codex_cli", base_env)
    assert "OPENAI_API_KEY" not in codex_env
    assert "ANTHROPIC_API_KEY" not in codex_env
    assert "ANTHROPIC_MODEL" not in codex_env
    assert codex_env["CODEX_HOME"] == "/tmp/codex-home"

    claude_env = _clean_env_for_provider("claude_cli", base_env)
    assert "ANTHROPIC_API_KEY" not in claude_env
    assert claude_env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"


def test_cli_argument_builders_use_noninteractive_structured_modes():
    codex_args = _build_codex_args(
        model="gpt-5.5",
        workdir="/repo",
        effort="xhigh",
        service_tier="fast",
        output_path="/tmp/out.txt",
        schema_path="/tmp/schema.json",
    )
    assert codex_args[:2] == ["exec", "--cd"]
    assert "--sandbox" in codex_args and "read-only" in codex_args
    assert "--output-last-message" in codex_args
    assert "--output-schema" in codex_args
    assert "model_reasoning_effort=\"xhigh\"" in codex_args
    assert "service_tier=\"fast\"" in codex_args
    assert codex_args[-1] == "-"

    claude_args = _build_claude_args(
        model="claude-opus-4-8",
        effort="xhigh",
        output_format="json",
        schema={"type": "object"},
    )
    assert claude_args[:3] == ["-p", "--output-format", "json"]
    assert "--json-schema" in claude_args
    assert "--tools" in claude_args
    assert "--no-session-persistence" in claude_args
    assert "--model" in claude_args and "claude-opus-4-8" in claude_args
    assert "--effort" in claude_args and "xhigh" in claude_args


def test_structured_binding_coerces_payload_to_pydantic(monkeypatch):
    class Pick(BaseModel):
        decision: str

    monkeypatch.setattr(
        LocalCliChatModel,
        "_run_structured_prompt",
        lambda self, prompt, schema: {"decision": "BUY"},
    )
    llm = LocalCliChatModel(
        provider="claude_cli",
        model_name="claude-opus-4-8",
        command="claude",
    )

    result = llm.with_structured_output(Pick).invoke("choose")

    assert isinstance(result, Pick)
    assert result.decision == "BUY"


def test_tool_binding_returns_langchain_tool_calls(monkeypatch):
    class Tool:
        name = "get_news"
        description = "Fetch news"
        args = {"query": {"type": "string"}}

    monkeypatch.setattr(
        LocalCliChatModel,
        "_run_structured_prompt",
        lambda self, prompt, schema: {
            "tool_calls": [{"name": "get_news", "args": {"query": "AAPL"}}],
            "final": None,
        },
    )
    llm = LocalCliChatModel(
        provider="codex_cli",
        model_name="gpt-5.5",
        command="codex",
    )

    message = llm.bind_tools([Tool()]).invoke("need data")

    assert message.content == ""
    assert message.tool_calls[0]["name"] == "get_news"
    assert message.tool_calls[0]["args"] == {"query": "AAPL"}
    assert message.tool_calls[0]["id"].startswith("call_")


def test_structured_output_unwraps_cli_json_shapes():
    assert _unwrap_structured_payload({"structured_output": {"x": 1}}) == {"x": 1}
    assert _unwrap_structured_payload({"result": "{\"x\": 2}"}) == {"x": 2}
    assert _loads_json_object("```json\n{\"x\": 3}\n```") == {"x": 3}
