"""
Agent 1 — Reasoning brain.  Model: Fanar-C-2-27B  (OpenAI-compatible chat).

This is the decision-maker: it understands the Arabic request and emits the next
JSON action (perceive / deliver / say / ask). Driven by the orchestrator loop.
"""
import json

import requests

from agent.fanar_base import FANAR_API_KEY, FANAR_BASE_URL, FANAR_MODEL, FanarError


class FanarClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key if api_key is not None else FANAR_API_KEY
        self.base_url = (base_url or FANAR_BASE_URL).rstrip("/")
        self.model = model or FANAR_MODEL
        if not self.api_key:
            raise FanarError("FANAR_API_KEY is not set. Run: export FANAR_API_KEY=...")

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        """Return the assistant message dict (may contain 'tool_calls')."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise FanarError(f"HTTP {resp.status_code}: {resp.text[:600]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise FanarError(f"Unexpected response shape: {json.dumps(data)[:600]}")
