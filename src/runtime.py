from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .local_runtime_backend import engine, get_agents, wrap_message


@dataclass
class RuntimeBundle:
    agent: object
    personas: object | None = None


def load_runtime(args) -> RuntimeBundle:
    agent, personas = get_agents(args)
    return RuntimeBundle(agent=agent, personas=personas)


def close_runtime(runtime: RuntimeBundle | None) -> None:
    if runtime is None:
        return
    agent = getattr(runtime, "agent", None)
    if agent is None:
        return
    shutdown = getattr(agent, "shutdown", None)
    if callable(shutdown):
        shutdown()


def run_prompts(runtime: RuntimeBundle, prompts: Iterable[tuple[str, object]], system_prompt: str | None = None) -> List[str]:
    prompt_items = list(prompts)
    for alias, prompt in prompt_items:
        if not isinstance(prompt, (str, dict)):
            raise TypeError(f"Prompt {alias!r} must be a string or message payload, got {type(prompt).__name__}.")
    set_trace_aliases = getattr(runtime.agent, "set_trace_aliases", None)
    if callable(set_trace_aliases):
        set_trace_aliases([str(alias) for alias, _ in prompt_items])
    wrapped = [wrap_message(None, alias, prompt, system_prompt) for alias, prompt in prompt_items]
    return engine(wrapped, runtime.agent, len(wrapped))
