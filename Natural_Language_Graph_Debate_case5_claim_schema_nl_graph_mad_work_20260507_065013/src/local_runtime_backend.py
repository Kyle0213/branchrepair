from __future__ import annotations

import os
import gc
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None


MODEL_DIRS = {
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
}


def _resolve_model_path(args):
    if args.model in MODEL_DIRS:
        return MODEL_DIRS[args.model]
    candidate = Path(args.model)
    if candidate.exists():
        return str(candidate)
    if getattr(args, "model_dir", None):
        candidate = Path(args.model_dir) / args.model
        if candidate.exists():
            return str(candidate)
    if "/" in args.model:
        return args.model
    raise KeyError(f"Unknown model '{args.model}'. Add it to MODEL_DIRS or pass a full model path.")


class VllmWrapper:
    def __init__(self, args, model_name_or_path):
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM is not installed. Install it or run with a supported local backend.")
        if getattr(args, "token", "") and "HUGGINGFACE_HUB_TOKEN" not in os.environ:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = args.token
        self.name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=getattr(args, "token", ""))
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vllm_llm = LLM(
            model=model_name_or_path,
            download_dir=getattr(args, "model_dir", None),
            dtype="auto",
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.75),
            max_model_len=getattr(args, "max_model_len", None),
        )
        self.runtime_sampling_config = _runtime_sampling_config(args, model_name_or_path)
        self.sampling_params = SamplingParams(
            **self.runtime_sampling_config,
            max_tokens=getattr(args, "max_new_tokens", 1024),
        )

    def shutdown(self) -> None:
        llm = getattr(self, "vllm_llm", None)
        if llm is None:
            return
        try:
            llm_engine = getattr(llm, "llm_engine", None)
            engine_core = getattr(llm_engine, "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown()
        except Exception:
            pass
        try:
            model_executor = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
            if model_executor is not None and hasattr(model_executor, "shutdown"):
                model_executor.shutdown()
        except Exception:
            pass
        self.vllm_llm = None
        gc.collect()


class OpenAICompatibleWrapper:
    def __init__(self, args):
        self.name = getattr(args, "api_model", None) or getattr(args, "model", "deepseek-chat")
        self.base_url = str(getattr(args, "api_base_url", "") or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.beta_base_url = str(getattr(args, "api_beta_base_url", "") or "").rstrip("/")
        self.api_key = str(getattr(args, "api_key", "") or os.environ.get("OPENAI_API_KEY", ""))
        if not self.base_url:
            raise ValueError("--api_base_url or OPENAI_BASE_URL is required for --runtime_backend openai_api")
        if not self.beta_base_url:
            self.beta_base_url = self.base_url if self.base_url.endswith("/beta") else f"{self.base_url}/beta"
        if not self.api_key:
            raise ValueError("--api_key or OPENAI_API_KEY is required for --runtime_backend openai_api")
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 1024) or 1024)
        self.runtime_sampling_config = _runtime_sampling_config(args, self.name)
        self.timeout = float(getattr(args, "api_timeout", 120) or 120)
        self.max_retries = int(getattr(args, "api_max_retries", 3) or 3)
        self.save_api_trace = bool(getattr(args, "save_api_trace", False))
        self.api_trace: list[dict] = []
        self._pending_trace_aliases: list[str] = []
        self.session = requests.Session()
        self.session.trust_env = False

    def set_trace_aliases(self, aliases: list[str]) -> None:
        self._pending_trace_aliases = [str(alias) for alias in aliases]

    def pop_api_trace(self) -> list[dict]:
        trace = list(self.api_trace)
        self.api_trace.clear()
        return trace

    def _next_trace_alias(self) -> str:
        if self._pending_trace_aliases:
            return self._pending_trace_aliases.pop(0)
        return ""

    def shutdown(self) -> None:
        close = getattr(self.session, "close", None)
        if callable(close):
            close()

    def chat(self, messages: list[dict], stop_sequences=None, *, prefix_completion: bool = False) -> str:
        trace_alias = self._next_trace_alias()
        payload = {
            "model": self.name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
        }
        for key in ("temperature", "top_p"):
            if key in self.runtime_sampling_config:
                payload[key] = self.runtime_sampling_config[key]
        if stop_sequences:
            payload["stop"] = stop_sequences
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        base_url = self.beta_base_url if prefix_completion else self.base_url
        url = f"{base_url}/chat/completions"
        last_error = None
        trace_record = None
        if self.save_api_trace:
            trace_record = {
                "alias": trace_alias,
                "stage": _stage_from_alias(trace_alias),
                "model": self.name,
                "url_kind": "beta" if prefix_completion else "standard",
                "prefix_completion": bool(prefix_completion),
                "max_tokens": self.max_new_tokens,
                "sampling": {
                    key: payload[key]
                    for key in ("temperature", "top_p")
                    if key in payload
                },
                "stop": list(stop_sequences) if stop_sequences else None,
                "messages": messages,
                "attempts": [],
            }
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                latency = time.time() - started
                attempt_record = None
                if trace_record is not None:
                    attempt_record = {
                        "attempt": attempt + 1,
                        "status_code": response.status_code,
                        "latency_sec": round(latency, 4),
                    }
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    if attempt_record is not None:
                        attempt_record["retryable"] = True
                        trace_record["attempts"].append(attempt_record)
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if response.status_code >= 400:
                    detail = response.text[:2000]
                    try:
                        detail = json.dumps(response.json(), ensure_ascii=False)[:2000]
                    except Exception:
                        pass
                    if attempt_record is not None:
                        attempt_record["error"] = detail
                        trace_record["attempts"].append(attempt_record)
                    raise RuntimeError(f"HTTP {response.status_code} from {url}: {detail}")
                data = response.json()
                choice = data["choices"][0]
                content = str(choice["message"].get("content") or "")
                if attempt_record is not None:
                    attempt_record["finish_reason"] = choice.get("finish_reason")
                    attempt_record["usage"] = data.get("usage")
                    trace_record["attempts"].append(attempt_record)
                    trace_record["output"] = content
                    trace_record["output_chars"] = len(content)
                    self.api_trace.append(trace_record)
                return content
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        if trace_record is not None:
            trace_record["error"] = str(last_error)
            self.api_trace.append(trace_record)
        raise RuntimeError(f"OpenAI-compatible API call failed after {self.max_retries + 1} attempts: {last_error}")


def _stage_from_alias(alias: str) -> str:
    text = str(alias or "")
    suffix = text.rsplit(":", 1)[-1]
    if suffix in {
        "normalize",
        "merge",
        "audit",
        "prefix_repair",
        "analysis",
        "resolve",
        "pair_analysis",
        "pair_resolve",
        "revise",
    }:
        return suffix
    if suffix.endswith("analysis"):
        return "analysis"
    if suffix.endswith("resolve"):
        return "resolve"
    if suffix.endswith("revise"):
        return "revise"
    return suffix or "unknown"


def _merge_system_prompt(system_prompt, persona_text=None):
    parts = []
    for text in (system_prompt, persona_text):
        if text and str(text).strip():
            parts.append(str(text).strip())
    return "\n\n".join(parts) if parts else None


def _is_chat_message(value) -> bool:
    return isinstance(value, dict) and "role" in value and "content" in value


def _with_system_message(messages: list[dict], system_text: str | None):
    if system_text:
        return [{"role": "system", "content": system_text}] + messages
    return messages


def wrap_message(personas, agent, msg, system_prompt=None):
    persona_text = None
    if personas is not None:
        persona_key = agent.split("__")[-2] if "__" in agent else agent
        persona_text = personas.get(persona_key) or personas.get("None")
    if isinstance(msg, dict) and "messages" in msg:
        payload_system_prompt = msg["system_prompt"] if "system_prompt" in msg else system_prompt
        system_text = _merge_system_prompt(payload_system_prompt, persona_text)
        messages = list(msg.get("messages") or [])
        return {
            "messages": _with_system_message(messages, system_text),
            "continue_final_message": bool(msg.get("continue_final_message")),
        }
    system_text = _merge_system_prompt(system_prompt, persona_text)
    if _is_chat_message(msg):
        return _with_system_message([msg], system_text)
    if isinstance(msg, list) and all(_is_chat_message(item) for item in msg):
        return _with_system_message(list(msg), system_text)
    if system_text:
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": msg},
        ]
    return {"role": "user", "content": msg}


def _get_context_limit(current_agent):
    candidates = []
    tokenizer = getattr(current_agent, "tokenizer", None)
    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < 10**6:
        candidates.append(max_len)
    if hasattr(current_agent, "vllm_llm"):
        try:
            max_len = int(current_agent.vllm_llm.llm_engine.model_config.max_model_len)
            if 0 < max_len < 10**6:
                candidates.append(max_len)
        except Exception:
            pass
    return min(candidates) if candidates else None


def _fit_prompt_to_context(current_agent, prompt, max_new_tokens):
    tokenizer = getattr(current_agent, "tokenizer", None)
    context_limit = _get_context_limit(current_agent)
    if tokenizer is None or context_limit is None:
        return prompt
    reserve_tokens = max(int(max_new_tokens or 0) + 32, 64)
    max_input_tokens = max(context_limit - reserve_tokens, 1024)
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_input_tokens:
        return prompt
    bridge = "\n\n[... prompt truncated to fit model context ...]\n\n"
    bridge_ids = tokenizer(bridge, add_special_tokens=False)["input_ids"]
    if len(bridge_ids) >= max_input_tokens:
        trimmed_ids = token_ids[-max_input_tokens:]
    else:
        head_budget = min(512, max_input_tokens // 4)
        tail_budget = max_input_tokens - len(bridge_ids) - head_budget
        if tail_budget < 128:
            tail_budget = max_input_tokens - len(bridge_ids)
            head_budget = 0
        trimmed_ids = token_ids[:head_budget] + bridge_ids + token_ids[-tail_budget:]
    return tokenizer.decode(trimmed_ids, skip_special_tokens=False)


def engine(messages, agent, num_agents=1, stop_sequences=None):
    del num_agents

    def _is_qwen3(current_agent):
        candidates = [
            getattr(current_agent, "name", ""),
            getattr(getattr(current_agent, "tokenizer", None), "name_or_path", ""),
        ]
        return any("qwen3" in str(name).lower() for name in candidates if name)

    def _apply_chat_template(current_agent, msgs, *, continue_final_message=False):
        add_generation_prompt = not continue_final_message
        if _is_qwen3(current_agent):
            enable_thinking = os.environ.get("QWEN3_ENABLE_THINKING", "").strip().lower() in {"1", "true", "yes", "on"}
            return current_agent.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                enable_thinking=enable_thinking,
            )
        return current_agent.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )

    prompt_specs = []
    for item in messages:
        if isinstance(item, dict) and "messages" in item:
            prompt_specs.append((list(item.get("messages") or []), bool(item.get("continue_final_message"))))
        elif isinstance(item, list):
            prompt_specs.append((item, False))
        else:
            prompt_specs.append(([item], False))
    if hasattr(agent, "chat") and not hasattr(agent, "vllm_llm"):
        api_outputs = []
        for msgs, continue_final_message in prompt_specs:
            if continue_final_message and msgs and msgs[-1].get("role") == "assistant":
                msgs = list(msgs)
                msgs[-1] = dict(msgs[-1])
                msgs[-1]["prefix"] = True
                api_outputs.append(agent.chat(msgs, stop_sequences=stop_sequences, prefix_completion=True))
                continue
            api_outputs.append(agent.chat(msgs, stop_sequences=stop_sequences))
        return api_outputs

    prompts = [
        _apply_chat_template(agent, msgs, continue_final_message=continue_final_message)
        for msgs, continue_final_message in prompt_specs
    ]
    if not hasattr(agent, "vllm_llm"):
        raise NotImplementedError("Standalone Natural_Language_Graph_Debate currently supports vLLM runtime only.")

    if _is_qwen3(agent):
        sampling_kwargs = dict(getattr(agent, "runtime_sampling_config", None) or _qwen3_sampling_params_kwargs())
        sampling_kwargs["max_tokens"] = getattr(getattr(agent, "sampling_params", None), "max_tokens", 1024)
        if stop_sequences:
            sampling_kwargs["stop"] = stop_sequences
        sampling_params = SamplingParams(**sampling_kwargs)
    elif stop_sequences:
        sampling_kwargs = dict(getattr(agent, "runtime_sampling_config", None) or {"temperature": 1.0, "top_p": 0.9})
        sampling_kwargs["max_tokens"] = getattr(getattr(agent, "sampling_params", None), "max_tokens", 1024)
        sampling_kwargs["stop"] = stop_sequences
        sampling_params = SamplingParams(
            **sampling_kwargs,
        )
    else:
        sampling_params = agent.sampling_params

    max_new_tokens = getattr(sampling_params, "max_tokens", 1024)
    prompts = [_fit_prompt_to_context(agent, prompt, max_new_tokens) for prompt in prompts]
    outputs = agent.vllm_llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def _is_qwen3_model_name(model_name_or_path: str) -> bool:
    return "qwen3" in str(model_name_or_path or "").lower()


def _is_llama_model_name(model_name_or_path: str) -> bool:
    lowered = str(model_name_or_path or "").lower()
    return "llama" in lowered


def _qwen3_sampling_params_kwargs():
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def _runtime_sampling_config(args, model_name_or_path: str) -> dict:
    if _is_qwen3_model_name(model_name_or_path):
        defaults = _qwen3_sampling_params_kwargs()
    elif _is_llama_model_name(model_name_or_path):
        defaults = {"temperature": 0.6, "top_p": 0.9}
    else:
        defaults = {"temperature": 1.0, "top_p": 0.9}
    config = dict(defaults)
    if getattr(args, "temperature", None) is not None:
        config["temperature"] = float(args.temperature)
    if getattr(args, "top_p", None) is not None:
        config["top_p"] = float(args.top_p)
    if getattr(args, "top_k", None) is not None:
        config["top_k"] = int(args.top_k)
    if getattr(args, "min_p", None) is not None:
        config["min_p"] = float(args.min_p)
    return config


def get_agents(args, peft_path=None):
    del peft_path
    if str(getattr(args, "runtime_backend", "") or "").strip().lower() in {"openai_api", "api"}:
        return OpenAICompatibleWrapper(args), {"None": ""}
    if not getattr(args, "use_vllm", True):
        raise NotImplementedError("Standalone Natural_Language_Graph_Debate currently supports --use_vllm only.")
    agent = VllmWrapper(args, _resolve_model_path(args))
    if agent.tokenizer.pad_token is None:
        agent.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    personas = {"None": ""}
    return agent, personas
