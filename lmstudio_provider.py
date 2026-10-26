"""OpenAI-compatible Chat Completions API for a local LM Studio server."""

import json
import os

import openai

from llm import ResponseError, ToolCall, system_prompt

BASE_URL = "http://localhost:1234/v1"
MODEL = "local-model"


class Provider:
    Unreachable = openai.APIError
    ConfigurationError = openai.OpenAIError

    def __init__(self, model=None):
        self.model = model or MODEL
        self.client = openai.OpenAI(
            base_url=os.getenv("LM_STUDIO_BASE_URL", BASE_URL),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        )

    def get_next_answer(self, conversation, tools, yolo=False):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt(yolo)},
                *conversation,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in tools
            ],
            tool_choice="auto",
        )
        if not response.choices:
            raise ResponseError("LM Studio returned no response choices")
        if response.choices[0].finish_reason in ("length", "content_filter"):
            raise ResponseError(
                f"LM Studio turn did not complete: {response.choices[0].finish_reason}"
            )
        return response

    def record_answer(self, conversation, message):
        conversation.append(
            message.choices[0].message.model_dump(exclude_none=True)
        )

    def text_of(self, message):
        return message.choices[0].message.content or ""

    def tool_calls_of(self, message):
        calls = []
        for item in message.choices[0].message.tool_calls or []:
            try:
                arguments = json.loads(item.function.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                call = ToolCall(item.id, item.function.name, arguments)
            except (ValueError, TypeError) as e:
                call = ToolCall(
                    item.id,
                    item.function.name,
                    {},
                    f"Invalid tool arguments: {e}",
                )
            calls.append(call)
        return calls

    def add_results(self, conversation, results):
        conversation.extend(
            {
                "role": "tool",
                "tool_call_id": result["id"],
                "content": ("Error: " if result.get("is_error") else "")
                + result["content"],
            }
            for result in results
        )
