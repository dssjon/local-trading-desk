"""LangChain-compatible clients backed by local Codex and Claude CLIs.

These providers are intentionally API-keyless from TradingAgents' point of
view. Authentication belongs to the already logged-in local CLI, and API-key
environment variables are stripped from subprocesses so the CLIs use their
subscription/OAuth auth paths instead of billable API keys.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from .base_client import BaseLLMClient
from .validators import validate_model

_DEFAULT_TIMEOUT_SECONDS = 600
_DEFAULT_CODEX_MODEL = "gpt-5.5"
_DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"
_DEFAULT_EFFORT = "xhigh"
_DEFAULT_CODEX_SERVICE_TIER = "fast"

_CODEX_STRIP_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_OAUTH_TOKEN",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_THREAD_ID",
    "OPENAI_API_BASE",
    "OPENAI_API_HOST",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT",
    "ZMX_SESSION",
}

_CLAUDE_STRIP_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
    "ZMX_SESSION",
}

_TOOL_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args_json": {
                        "type": "string",
                        "description": "JSON object string containing the tool arguments.",
                    },
                },
                "required": ["name", "args_json"],
                "additionalProperties": False,
            },
        },
        "final": {"type": ["string", "null"]},
    },
    "required": ["tool_calls", "final"],
    "additionalProperties": False,
}


def _clean_env_for_provider(provider: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment that forces local CLI auth instead of API-key auth."""
    env = dict(base_env or os.environ)
    strip = _CODEX_STRIP_ENV if provider == "codex_cli" else _CLAUDE_STRIP_ENV
    for key in strip:
        env.pop(key, None)
    env.setdefault("NO_COLOR", "1")
    return env


def _build_codex_args(
    *,
    model: str | None,
    workdir: str,
    effort: str | None,
    service_tier: str | None,
    output_path: str,
    schema_path: str | None = None,
) -> list[str]:
    args = [
        "exec",
        "--cd",
        workdir,
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "--output-last-message",
        output_path,
    ]
    if model:
        args.extend(["-m", model])
    if effort:
        args.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
    if service_tier:
        args.extend(["-c", f"service_tier={json.dumps(service_tier)}"])
    if schema_path:
        args.extend(["--output-schema", schema_path])
    args.append("-")
    return args


def _build_claude_args(
    *,
    model: str | None,
    effort: str | None,
    output_format: str,
    schema: dict[str, Any] | None = None,
) -> list[str]:
    args = ["-p", "--output-format", output_format, "--tools", "", "--no-session-persistence"]
    if model:
        args.extend(["--model", model])
    if effort:
        args.extend(["--effort", effort])
    if schema is not None:
        args.extend(["--json-schema", json.dumps(schema)])
    return args


def _run_process(
    command: str,
    args: list[str],
    prompt: str,
    *,
    env: dict[str, str],
    cwd: str | None,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [command, *args],
            input=prompt,
            text=True,
            capture_output=True,
            env=env,
            cwd=cwd,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Local CLI command {command!r} was not found. Install it or set the "
            "matching TRADINGAGENTS_*_BIN environment variable."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Local CLI command {command!r} exceeded {timeout_seconds} seconds."
        ) from exc


def _require_success(process: subprocess.CompletedProcess[str], command: str) -> None:
    if process.returncode == 0:
        return
    stderr = (process.stderr or process.stdout or "").strip()
    if len(stderr) > 2000:
        stderr = stderr[-2000:]
    raise RuntimeError(f"Local CLI command {command!r} failed with exit {process.returncode}: {stderr}")


def _extract_codex_last_message(stdout: str) -> str:
    """Best-effort fallback when --output-last-message did not write a file."""
    last_text = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        if event.get("type") == "item.completed" and isinstance(item.get("text"), str):
            last_text = item["text"]
        elif event.get("type") == "item.delta" and isinstance(item.get("delta"), str):
            last_text += item["delta"]
    return last_text


def _loads_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty structured response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1).strip())

    start = min([idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0], default=-1)
    if start < 0:
        raise
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        raise
    return json.loads(stripped[start : end + 1])


def _unwrap_structured_payload(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    for key in ("structured_output", "parsed"):
        if key in raw:
            return raw[key]
    if "result" in raw:
        result = raw["result"]
        if isinstance(result, str):
            try:
                return _loads_json_object(result)
            except Exception:
                return result
        return result
    return raw


def _schema_to_json_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if hasattr(schema, "schema"):
        return schema.schema()
    raise TypeError(f"Unsupported structured output schema: {schema!r}")


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a Codex/OpenAI strict-compatible JSON schema copy."""
    if not isinstance(schema, dict):
        return schema

    strict: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            strict[key] = {
                name: _strict_json_schema(child) if isinstance(child, dict) else child
                for name, child in value.items()
            }
        elif key in {"items", "additionalProperties"} and isinstance(value, dict):
            strict[key] = _strict_json_schema(value)
        elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, list):
            strict[key] = [
                _strict_json_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            strict[key] = value

    if strict.get("type") == "object" or "properties" in strict:
        strict["additionalProperties"] = False
    return strict


def _coerce_to_schema(schema: Any, payload: Any) -> Any:
    if isinstance(schema, dict):
        return payload
    if hasattr(schema, "model_validate"):
        return schema.model_validate(payload)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(payload)
    return payload


def _input_to_messages(input_: Any) -> list[Any]:
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None)
    if role:
        return str(role)
    return message.__class__.__name__.removesuffix("Message").lower()


def _render_messages(input_: Any) -> str:
    messages = _input_to_messages(input_)
    if not messages:
        return str(input_)

    rendered: list[str] = []
    for message in messages:
        role = _message_role(message).upper()
        rendered.append(f"{role}: {_content_to_text(getattr(message, 'content', message))}")
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            rendered.append(
                "ASSISTANT TOOL CALLS: "
                + json.dumps(tool_calls, ensure_ascii=False, default=str)
            )
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            rendered.append(f"TOOL CALL ID: {tool_call_id}")
    return "\n\n".join(rendered)


def _tool_to_spec(tool: Any) -> dict[str, Any]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            schema = _schema_to_json_schema(args_schema)
        except Exception:
            schema = getattr(tool, "args", {})
    else:
        schema = getattr(tool, "args", {})

    return {
        "name": getattr(tool, "name", tool.__class__.__name__),
        "description": getattr(tool, "description", "") or "",
        "args_schema": schema,
    }


def _tool_planning_prompt(messages: Any, tools: list[Any]) -> str:
    tool_specs = [_tool_to_spec(tool) for tool in tools]
    return (
        "You are the tool-calling adapter for TradingAgents. Decide whether to call "
        "one or more available tools or produce the final analyst response.\n\n"
        "Rules:\n"
        "- Return only data matching the provided JSON schema.\n"
        "- Use tool_calls when tool data is needed. Do not invent data.\n"
        "- Do not batch dependent calls. If one tool needs output from another, request only the first call.\n"
        "- If existing tool results are sufficient, set tool_calls to [] and put the complete final report in final.\n"
        "- Tool names and argument keys must exactly match the available tool specs.\n\n"
        "- For each tool call, args_json must be a JSON object encoded as a string, e.g. {\"ticker\":\"AAPL\"}.\n\n"
        f"Available tools:\n{json.dumps(tool_specs, indent=2, ensure_ascii=False, default=str)}\n\n"
        f"Conversation:\n{_render_messages(messages)}"
    )


def _plain_prompt(messages: Any) -> str:
    return (
        "Respond to the following TradingAgents prompt. Do not run shell commands, "
        "inspect local files, or use external tools; use only the provided prompt "
        "content and return the answer text.\n\n"
        f"{_render_messages(messages)}"
    )


class _LocalCliStructuredBinding(Runnable):
    def __init__(self, model: LocalCliChatModel, schema: Any):
        self.model = model
        self.schema = schema

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.model._invoke_structured(input, self.schema)


class _LocalCliToolBinding(Runnable):
    def __init__(self, model: LocalCliChatModel, tools: list[Any]):
        self.model = model
        self.tools = tools

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        decision = self.model._invoke_tool_decision(input, self.tools)
        allowed_names = {getattr(tool, "name", tool.__class__.__name__) for tool in self.tools}

        tool_calls = []
        invalid_names = []
        for call in decision.get("tool_calls") or []:
            name = call.get("name")
            args = call.get("args")
            if args is None and isinstance(call.get("args_json"), str):
                try:
                    args = _loads_json_object(call["args_json"])
                except Exception:
                    args = {}
            args = args or {}
            if name not in allowed_names:
                invalid_names.append(name)
                continue
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                }
            )

        if tool_calls:
            return AIMessage(content="", tool_calls=tool_calls)

        final = decision.get("final")
        if not final and invalid_names:
            final = f"Unable to call unavailable tool(s): {', '.join(map(str, invalid_names))}."
        return AIMessage(content=final or "")


class LocalCliChatModel(BaseChatModel):
    """Minimal chat model that shells out to Codex CLI or Claude CLI."""

    provider: str
    model_name: str
    command: str
    effort: str | None = None
    service_tier: str | None = None
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    workdir: str | None = None

    @property
    def _llm_type(self) -> str:
        return self.provider

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "command": self.command,
            "effort": self.effort,
            "service_tier": self.service_tier,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._run_text(_plain_prompt(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Runnable:
        return _LocalCliToolBinding(self, list(tools))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return _LocalCliStructuredBinding(self, schema)

    def _run_text(self, prompt: str) -> str:
        if self.provider == "codex_cli":
            return self._run_codex(prompt)
        return self._run_claude(prompt, output_format="text")

    def _invoke_structured(self, input_: Any, schema: Any) -> Any:
        json_schema = _schema_to_json_schema(schema)
        prompt = (
            "Return only a response matching the supplied JSON schema for this "
            "TradingAgents prompt.\n\n"
            f"{_render_messages(input_)}"
        )
        payload = self._run_structured_prompt(prompt, json_schema)
        return _coerce_to_schema(schema, payload)

    def _invoke_tool_decision(self, input_: Any, tools: list[Any]) -> dict[str, Any]:
        payload = self._run_structured_prompt(
            _tool_planning_prompt(input_, tools),
            _TOOL_DECISION_SCHEMA,
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Tool planner returned non-object payload: {payload!r}")
        payload.setdefault("tool_calls", [])
        payload.setdefault("final", None)
        return payload

    def _run_structured_prompt(self, prompt: str, schema: dict[str, Any]) -> Any:
        schema = _strict_json_schema(schema)
        if self.provider == "codex_cli":
            raw = self._run_codex(prompt, schema=schema)
        else:
            raw = self._run_claude(prompt, output_format="json", schema=schema)
        return _unwrap_structured_payload(_loads_json_object(raw))

    def _run_codex(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        workdir = self.workdir or os.getcwd()
        env = _clean_env_for_provider("codex_cli")
        with tempfile.TemporaryDirectory(prefix="tradingagents-codex-") as tmpdir:
            output_path = str(Path(tmpdir) / "last-message.txt")
            schema_path = None
            if schema is not None:
                schema_file = Path(tmpdir) / "schema.json"
                schema_file.write_text(json.dumps(schema), encoding="utf-8")
                schema_path = str(schema_file)

            args = _build_codex_args(
                model=self.model_name or _DEFAULT_CODEX_MODEL,
                workdir=workdir,
                effort=self.effort,
                service_tier=self.service_tier,
                output_path=output_path,
                schema_path=schema_path,
            )
            process = _run_process(
                self.command,
                args,
                prompt,
                env=env,
                cwd=workdir,
                timeout_seconds=self.timeout_seconds,
            )
            _require_success(process, self.command)

            output_file = Path(output_path)
            if output_file.exists():
                text = output_file.read_text(encoding="utf-8").strip()
                if text:
                    return text
            return _extract_codex_last_message(process.stdout).strip()

    def _run_claude(
        self,
        prompt: str,
        *,
        output_format: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        env = _clean_env_for_provider("claude_cli")
        args = _build_claude_args(
            model=self.model_name or _DEFAULT_CLAUDE_MODEL,
            effort=self.effort,
            output_format=output_format,
            schema=schema,
        )
        process = _run_process(
            self.command,
            args,
            prompt,
            env=env,
            cwd=self.workdir or os.getcwd(),
            timeout_seconds=self.timeout_seconds,
        )
        _require_success(process, self.command)
        return (process.stdout or "").strip()


class LocalCliClient(BaseLLMClient):
    """Client for local Codex CLI and Claude CLI providers."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        provider: str,
        **kwargs: Any,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        if self.provider == "codex_cli":
            command = (
                self.kwargs.get("command")
                or os.environ.get("TRADINGAGENTS_CODEX_BIN")
                or os.environ.get("CODEX_BIN")
                or "codex"
            )
            effort = (
                self.kwargs.get("effort")
                or os.environ.get("TRADINGAGENTS_CODEX_EFFORT")
                or os.environ.get("CODEX_REASONING_EFFORT")
                or _DEFAULT_EFFORT
            )
            service_tier = (
                self.kwargs.get("service_tier")
                or os.environ.get("TRADINGAGENTS_CODEX_SERVICE_TIER")
                or os.environ.get("CODEX_SERVICE_TIER")
                or _DEFAULT_CODEX_SERVICE_TIER
            )
        elif self.provider == "claude_cli":
            command = (
                self.kwargs.get("command")
                or os.environ.get("TRADINGAGENTS_CLAUDE_BIN")
                or os.environ.get("CLAUDE_BIN")
                or "claude"
            )
            effort = (
                self.kwargs.get("effort")
                or os.environ.get("TRADINGAGENTS_CLAUDE_EFFORT")
                or os.environ.get("CLAUDE_CODE_EFFORT_LEVEL")
                or _DEFAULT_EFFORT
            )
            service_tier = None
        else:
            raise ValueError(f"Unsupported local CLI provider: {self.provider}")

        timeout_seconds = int(
            self.kwargs.get("timeout")
            or os.environ.get("TRADINGAGENTS_CLI_TIMEOUT_SECONDS")
            or _DEFAULT_TIMEOUT_SECONDS
        )
        workdir = self.kwargs.get("workdir") or os.environ.get("TRADINGAGENTS_CLI_WORKDIR")

        passthrough: dict[str, Any] = {}
        if "callbacks" in self.kwargs:
            passthrough["callbacks"] = self.kwargs["callbacks"]

        return LocalCliChatModel(
            provider=self.provider,
            model_name=self.model,
            command=str(command),
            effort=str(effort) if effort else None,
            service_tier=str(service_tier) if service_tier else None,
            timeout_seconds=timeout_seconds,
            workdir=str(workdir) if workdir else None,
            **passthrough,
        )

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
