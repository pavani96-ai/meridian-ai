"""LangChain v1 middleware for the specialist agents.

Middleware wraps the agent loop (model call -> tool calls -> model call ...) without
touching the agent's prompt or tools. Think of it as airport security around the agent:
every request and every tool call passes through these checks, in list order.

    request ─► [PII redaction] ─► [model call limit] ─► [model retry] ─► the LLM
                                                                       │ wants a tool
    tool    ◄─ [tool call log] ◄─ [tool call limit] ◄─ [tool retry]  ◄─┘

Built-in middleware used here (from langchain.agents.middleware):
  * PIIMiddleware            - redacts e-mail addresses / card numbers before they reach the LLM
  * ModelCallLimitMiddleware - stops an agent that loops (cost guard)
  * ToolCallLimitMiddleware  - caps how many tools one audit phase can call
  * ModelRetryMiddleware     - retries the LLM on transient errors (429 / 503) with backoff
  * ToolRetryMiddleware      - retries flaky tools (e.g. the live FX-rate API)
Custom middleware written here:
  * PromptInjectionMiddleware - blocks known overrides in user/tool messages before model calls
  * log_tool_calls           - one structured log line per tool call: name, args, duration, outcome
"""

import time

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from config.settings import settings
from logger import GLOBAL_LOGGER as log
from exceptions import PromptInjectionError
from prompt_security import contains_prompt_injection


class PromptInjectionMiddleware(AgentMiddleware):
    """Stop known overrides in human/tool messages before each agent model call.

    Trusted system prompts are excluded. Detection is heuristic, not comprehensive.
    """

    def before_model(self, state, runtime):
        for message in state.get("messages", []):
            if getattr(message, "type", None) not in {"human", "tool"}:
                continue
            content = message.content
            texts = [content] if isinstance(content, str) else [
                block.get("text", "") for block in content if isinstance(block, dict)
            ]
            if any(contains_prompt_injection(text) for text in texts if isinstance(text, str)):
                log.warning("Blocked agent prompt injection", message_type=message.type)
                raise PromptInjectionError(
                    "Potential instruction override detected in request or tool evidence. "
                    "Audit stopped; review the content before retrying."
                )
        return None

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


class ToolCallLoggingMiddleware(AgentMiddleware):
    """One completion event per tool call, without logging sensitive argument values."""

    def _record(self, request, started, outcome):
        log.info(
            "Agent tool call",
            tool=request.tool_call["name"],
            args={key: "[REDACTED]" for key in request.tool_call.get("args", {})},
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            outcome=outcome,
        )

    def wrap_tool_call(self, request, handler):
        started = time.perf_counter()
        try:
            result = handler(request)
        except Exception:
            self._record(request, started, "exception")
            raise
        self._record(request, started, getattr(result, "status", "success"))
        return result

    async def awrap_tool_call(self, request, handler):
        started = time.perf_counter()
        try:
            result = await handler(request)
        except Exception:
            self._record(request, started, "exception")
            raise
        self._record(request, started, getattr(result, "status", "success"))
        return result


log_tool_calls = ToolCallLoggingMiddleware()


def build_agent_middleware() -> list[AgentMiddleware]:
    """Create fresh guards/retries and share the stateless tool logger.

    Retries handle raised exceptions; tools returning error text/dicts do not retry.
    Call limits apply to the agent loop, not LLM calls made inside tools.
    """
    return [
        PromptInjectionMiddleware(),
        PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_tool_results=True),
        PIIMiddleware("credit_card", strategy="redact", apply_to_input=True, apply_to_tool_results=True),
        ModelCallLimitMiddleware(run_limit=settings.agent_max_model_calls, exit_behavior="end"),
        log_tool_calls,
        ToolCallLimitMiddleware(run_limit=settings.agent_max_tool_calls, exit_behavior="continue"),
        ModelRetryMiddleware(
            max_retries=settings.agent_max_retries, initial_delay=1.0, on_failure="error"
        ),
        ToolRetryMiddleware(max_retries=settings.agent_max_retries, initial_delay=0.5),
    ]
