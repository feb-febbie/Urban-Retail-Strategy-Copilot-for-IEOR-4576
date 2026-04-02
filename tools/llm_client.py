"""
Unified LLM client with automatic provider detection and fallback.

Priority order (highest to lowest):
  1. Anthropic  (claude-sonnet-4-6)           — ANTHROPIC_API_KEY
  2. Groq       (llama-3.3-70b-versatile)     — GROQ_API_KEY       (free tier)
  3. Google     (gemini-2.0-flash)            — GOOGLE_API_KEY     (free tier)
  4. Ollama     (best locally available)      — no key needed, local

All providers return an Anthropic-shaped response so the agents need zero changes.
The response object always has:
    .content   → list of NormalizedTextBlock | NormalizedToolUseBlock
    .stop_reason → "end_turn" | "tool_use"
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import requests as _requests  # alias to avoid conflict with local `requests`


# ── Normalised response types (mimic Anthropic SDK objects) ───────────────────

@dataclass
class NormalizedTextBlock:
    text: str
    type: str = "text"


@dataclass
class NormalizedToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class NormalizedResponse:
    content: list
    stop_reason: str  # "end_turn" | "tool_use"


# ── Provider detection ────────────────────────────────────────────────────────

_PREFERRED_OLLAMA_MODELS = [
    # Best tool-use capability first
    "qwen2.5:72b", "qwen2.5:32b", "qwen2.5:14b", "qwen2.5:7b",
    "llama3.1:70b", "llama3.1:8b",
    "llama3.2:latest", "llama3.2:3b",
    "mistral:latest", "gemma3:latest",
]


def _pick_ollama_model(available: list[str]) -> Optional[str]:
    """Return the highest-priority Ollama model that is locally available."""
    available_lower = {m.lower() for m in available}
    for preferred in _PREFERRED_OLLAMA_MODELS:
        if preferred.lower() in available_lower:
            return preferred
    # Fall back to whatever is installed
    return available[0] if available else None


def _detect_provider() -> tuple[str, Any, str]:
    """
    Try providers in order and return (provider_name, client, model_id)
    for the first one that is reachable.
    Raises RuntimeError if none work.

    Priority:
      1. Anthropic  — best quality, paid
      2. Gemini     — Google-native, free tier; on Cloud Run uses ADC (no key needed)
      3. Groq       — free tier fallback
      4. Ollama     — local fallback
    """
    # ── 1. Anthropic ──────────────────────────────────────────────────────────
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and not key.startswith("your_") and len(key) > 20:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            print("[LLM] Using Anthropic claude-sonnet-4-6")
            return ("anthropic", client, "claude-sonnet-4-6")
        except Exception as exc:
            print(f"[LLM] Anthropic init failed: {exc}")

    # ── 2. Google Gemini ───────────────────────────────────────────────────────
    # Path A: explicit API key (local dev / AI Studio free tier)
    key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if key and not key.startswith("your_"):
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            print("[LLM] Using Gemini 2.0 Flash (API key)")
            return ("gemini", client, "gemini-2.0-flash")
        except Exception as exc:
            print(f"[LLM] Gemini (API key) init failed: {exc}")

    # Path B: Native google-genai SDK with vertexai=True — same approach as previous project.
    # Does NOT require Model Garden approval. Works on Cloud Run with ADC automatically.
    try:
        from google import genai as _genai
        _genai_client = _genai.Client(vertexai=True)
        # Probe with a minimal call to confirm the model is reachable
        _genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="hi",
        )
        print("[LLM] Using Gemini 2.0 Flash (native google-genai SDK, Vertex AI ADC)")
        return ("gemini_native", _genai_client, "gemini-2.0-flash")
    except Exception as exc:
        print(f"[LLM] Gemini native ADC not available: {exc}")

    # Path C: OpenAI-compat Vertex AI — requires Model Garden approval (fallback only)
    _VERTEX_MODELS = [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-1.5-flash-002",
        "google/gemini-1.5-flash-001",
    ]
    try:
        import google.auth
        import google.auth.transport.requests
        credentials, project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(google.auth.transport.requests.Request())
        from openai import OpenAI
        base_url = (
            f"https://us-central1-aiplatform.googleapis.com/v1beta1"
            f"/projects/{project}/locations/us-central1/endpoints/openapi"
        )
        client = OpenAI(api_key=credentials.token, base_url=base_url)
        for model in _VERTEX_MODELS:
            try:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                print(f"[LLM] Using Vertex AI ADC (OpenAI-compat) → {model} (project: {project})")
                return ("gemini", client, model)
            except Exception as model_exc:
                print(f"[LLM] Vertex AI model {model} unavailable: {model_exc}")
                continue
    except Exception as exc:
        print(f"[LLM] Gemini ADC (OpenAI-compat) not available: {exc}")

    # ── 3. Groq (free tier, OpenAI-compat) ────────────────────────────────────
    key = os.environ.get("GROQ_API_KEY", "")
    if key and not key.startswith("your_"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            # Try models in order of tool-use reliability
            for groq_model in [
                "llama3-groq-70b-8192-tool-use-preview",
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
            ]:
                try:
                    client.chat.completions.create(
                        model=groq_model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                    )
                    print(f"[LLM] Using Groq {groq_model} (free tier)")
                    return ("groq", client, groq_model)
                except Exception:
                    continue
        except Exception as exc:
            print(f"[LLM] Groq init failed: {exc}")

    # ── 4. Ollama (fully local, no key needed) ────────────────────────────────
    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            chosen = _pick_ollama_model(models)
            if chosen:
                from openai import OpenAI
                client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
                print(f"[LLM] Using Ollama model: {chosen}")
                return ("ollama", client, chosen)
            else:
                print("[LLM] Ollama running but no models installed. Run: ollama pull llama3.2")
    except Exception:
        pass  # Ollama not running — that's fine

    raise RuntimeError(
        "No LLM provider is available.\n"
        "Options (in order of preference):\n"
        "  1. Set ANTHROPIC_API_KEY in your .env  (paid)\n"
        "  2. Set GROQ_API_KEY in your .env        (free: console.groq.com)\n"
        "  3. Set GOOGLE_API_KEY in your .env      (free: aistudio.google.com)\n"
        "  4. Install Ollama: https://ollama.com   (local, no key)\n"
        "     Then run: ollama pull llama3.2"
    )


# ── Format converters ─────────────────────────────────────────────────────────

def _anthropic_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool schema → OpenAI function schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in (tools or [])
    ]


def _anthropic_messages_to_openai(messages: list[dict]) -> list[dict]:
    """
    Convert an Anthropic-format message list → OpenAI-format message list.

    Handles:
      - Plain string content
      - List content with text / tool_use / tool_result blocks
      - Anthropic SDK objects (with .type attribute) and plain dicts
    """
    out = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Detect tool_result blocks
                tool_results = [
                    c for c in content
                    if (isinstance(c, dict) and c.get("type") == "tool_result")
                ]
                if tool_results:
                    for tr in tool_results:
                        body = tr.get("content", "")
                        out.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": body if isinstance(body, str) else json.dumps(body),
                        })
                else:
                    # Normal multi-part user message — flatten to text
                    text = " ".join(
                        (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
                        for b in content
                    )
                    out.append({"role": "user", "content": text or str(content)})

        elif role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict] = []

                for block in content:
                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

                    if btype == "text":
                        txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                        if txt:
                            text_parts.append(txt)

                    elif btype == "tool_use":
                        bid = block.get("id") if isinstance(block, dict) else getattr(block, "id", "")
                        bname = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
                        binput = block.get("input", {}) if isinstance(block, dict) else getattr(block, "input", {})
                        tool_calls.append({
                            "id": bid,
                            "type": "function",
                            "function": {
                                "name": bname,
                                "arguments": json.dumps(binput),
                            },
                        })

                entry: dict = {"role": "assistant", "content": " ".join(text_parts) or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                out.append(entry)

    return out


def _openai_response_to_anthropic(response) -> NormalizedResponse:
    """Convert an OpenAI-compat response → NormalizedResponse (Anthropic-shaped)."""
    choice = response.choices[0]
    msg = choice.message
    blocks: list = []
    stop_reason = "end_turn"

    if msg.content:
        blocks.append(NormalizedTextBlock(text=msg.content))

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        stop_reason = "tool_use"
        for tc in msg.tool_calls:
            try:
                input_dict = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                input_dict = {"_raw": tc.function.arguments}
            blocks.append(NormalizedToolUseBlock(
                id=tc.id,
                name=tc.function.name,
                input=input_dict,
            ))

    if choice.finish_reason == "tool_calls":
        stop_reason = "tool_use"

    return NormalizedResponse(content=blocks, stop_reason=stop_reason)


def _anthropic_to_genai_tools(tools: list[dict]) -> list:
    """Convert Anthropic tool schema → google-genai Tool list."""
    from google.genai import types as _gt

    declarations = []
    for t in (tools or []):
        schema = t.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(
            _gt.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=schema,
            )
        )
    return [_gt.Tool(function_declarations=declarations)]


def _anthropic_messages_to_genai(messages: list[dict]) -> list:
    """Convert Anthropic message list → google-genai Content list."""
    from google.genai import types as _gt

    # Pre-scan: build tool_use_id → tool_name map (needed for FunctionResponse)
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype == "tool_use":
                    bid = block.get("id") if isinstance(block, dict) else getattr(block, "id", "")
                    bname = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
                    tool_id_to_name[bid] = bname

    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        genai_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append(_gt.Content(role=genai_role, parts=[_gt.Part(text=content)]))
            continue

        if not isinstance(content, list):
            continue

        # Check for tool_result blocks — these become user-role FunctionResponse parts
        tool_results = [
            b for b in content
            if (isinstance(b, dict) and b.get("type") == "tool_result")
        ]
        if tool_results:
            parts = []
            for tr in tool_results:
                tool_use_id = tr.get("tool_use_id", "")
                tool_name = tool_id_to_name.get(tool_use_id, tool_use_id)
                result_body = tr.get("content", "")
                if isinstance(result_body, list):
                    result_body = json.dumps(result_body)
                parts.append(_gt.Part(
                    function_response=_gt.FunctionResponse(
                        name=tool_name,
                        response={"result": result_body},
                    )
                ))
            if parts:
                contents.append(_gt.Content(role="user", parts=parts))
            continue

        # Normal content list (may mix text + tool_use blocks)
        parts = []
        for block in content:
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

            if btype == "text":
                txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                if txt:
                    parts.append(_gt.Part(text=txt))

            elif btype == "tool_use":
                bname = block.get("name") if isinstance(block, dict) else getattr(block, "name", "")
                binput = block.get("input", {}) if isinstance(block, dict) else getattr(block, "input", {})
                parts.append(_gt.Part(
                    function_call=_gt.FunctionCall(name=bname, args=binput)
                ))

        if parts:
            contents.append(_gt.Content(role=genai_role, parts=parts))

    return contents


def _genai_response_to_normalized(response) -> NormalizedResponse:
    """Convert a google-genai GenerateContentResponse → NormalizedResponse."""
    blocks: list = []
    stop_reason = "end_turn"

    if not response.candidates:
        return NormalizedResponse(content=blocks, stop_reason=stop_reason)

    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return NormalizedResponse(content=blocks, stop_reason=stop_reason)

    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text:
            blocks.append(NormalizedTextBlock(text=part.text))
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            stop_reason = "tool_use"
            blocks.append(NormalizedToolUseBlock(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=fc.name,
                input=dict(fc.args) if fc.args else {},
            ))

    return NormalizedResponse(content=blocks, stop_reason=stop_reason)


def _anthropic_response_to_normalized(response) -> NormalizedResponse:
    """Wrap an Anthropic SDK response as a NormalizedResponse."""
    blocks = []
    for block in response.content:
        if block.type == "text":
            blocks.append(NormalizedTextBlock(text=block.text))
        elif block.type == "tool_use":
            blocks.append(NormalizedToolUseBlock(
                id=block.id, name=block.name, input=block.input
            ))
    stop = response.stop_reason or "end_turn"
    return NormalizedResponse(content=blocks, stop_reason=stop)


# ── Public client class ───────────────────────────────────────────────────────

class UnifiedLLMClient:
    """
    Drop-in replacement for `anthropic.Anthropic()`.
    Detects the best available provider once at init time.
    Call `.messages_create()` the same way you would call
    `anthropic_client.messages.create()`.
    """

    def __init__(self):
        self.provider, self._client, self.model = _detect_provider()

    def messages_create(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        max_tokens: int = 4096,
        force_tool: Optional[str] = None,
    ) -> NormalizedResponse:
        """
        Unified message creation.
        Returns a NormalizedResponse regardless of which provider is active.

        force_tool: if set to a tool name, instructs the model to call that tool
                    immediately rather than replying with text (ANY mode).
                    Eliminates chatty preamble before tool calls.
        """
        if self.provider == "anthropic":
            return self._call_anthropic(system, messages, tools, max_tokens, force_tool)
        elif self.provider == "gemini_native":
            return self._call_genai(system, messages, tools, max_tokens, force_tool)
        else:
            return self._call_openai_compat(system, messages, tools, max_tokens, force_tool)

    # ── Anthropic path ─────────────────────────────────────────────────────

    def _call_anthropic(self, system, messages, tools, max_tokens, force_tool=None) -> NormalizedResponse:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        if force_tool and tools:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}

        raw = self._client.messages.create(**kwargs)
        return _anthropic_response_to_normalized(raw)

    # ── Native google-genai path (Vertex AI ADC) ──────────────────────────

    def _call_genai(self, system, messages, tools, max_tokens, force_tool=None) -> NormalizedResponse:
        from google.genai import types as _gt

        contents = _anthropic_messages_to_genai(messages)

        config_kwargs: dict = dict(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )
        if tools:
            config_kwargs["tools"] = _anthropic_to_genai_tools(tools)
            # Disable automatic function calling — we drive our own tool loop
            config_kwargs["automatic_function_calling"] = _gt.AutomaticFunctionCallingConfig(
                disable=True
            )
            if force_tool:
                # ANY mode: model must call a function, restricted to the named tool.
                # Equivalent to Anthropic's tool_choice={"type":"tool","name":force_tool}.
                config_kwargs["tool_config"] = _gt.ToolConfig(
                    function_calling_config=_gt.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[force_tool],
                    )
                )

        raw = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=_gt.GenerateContentConfig(**config_kwargs),
        )
        return _genai_response_to_normalized(raw)

    # ── OpenAI-compatible path (Groq / Gemini / Ollama) ───────────────────

    def _call_openai_compat(self, system, messages, tools, max_tokens, force_tool=None) -> NormalizedResponse:
        openai_msgs = [{"role": "system", "content": system}]
        openai_msgs += _anthropic_messages_to_openai(messages)

        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=openai_msgs,
        )
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)
            if force_tool:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
            else:
                kwargs["tool_choice"] = "auto"

        raw = self._client.chat.completions.create(**kwargs)
        return _openai_response_to_anthropic(raw)


# ── Module-level singleton ────────────────────────────────────────────────────

_instance: Optional[UnifiedLLMClient] = None


def get_llm_client() -> UnifiedLLMClient:
    """Return the module-level singleton, detecting provider on first call."""
    global _instance
    if _instance is None:
        _instance = UnifiedLLMClient()
    return _instance


def reset_llm_client():
    """Force re-detection on next call (useful for testing or key changes)."""
    global _instance
    _instance = None
