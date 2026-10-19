"""OpenAI Responses API with Sancho's own tools, not hosted tools."""

import json

import openai

from llm import ResponseError, ToolCall, system_prompt

MODEL = "gpt-5.1-codex"


class Provider:
    Unreachable = openai.APIError
    ConfigurationError = openai.OpenAIError

    def __init__(self, model=None):
        self.model = model or MODEL
        self.client = openai.OpenAI()

    def get_next_answer(self, conversation, tools, yolo=False):
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt(yolo),
            input=conversation,
            tools=[
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                    "strict": False,
                }
                for tool in tools
            ],
            store=False,
            include=["reasoning.encrypted_content"],
        )
        if response.status != "completed":
            detail = response.error or response.incomplete_details or response.status
            raise ResponseError(f"OpenAI turn did not complete: {detail}")
        return response

    def record_answer(self, conversation, message):
        # Reasoning items must travel with the function calls on subsequent requests.
        conversation.extend(message.output)

    def text_of(self, message):
        return message.output_text

    def tool_calls_of(self, message):
        calls = []
        for item in message.output:
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                call = ToolCall(item.call_id, item.name, arguments)
            except (ValueError, TypeError) as e:
                call = ToolCall(item.call_id, item.name, {}, f"Invalid tool arguments: {e}")
            calls.append(call)
        return calls

    def add_results(self, conversation, results):
        conversation.extend(
            {
                "type": "function_call_output",
                "call_id": result["id"],
                "output": ("Error: " if result.get("is_error") else "") + result["content"],
            }
            for result in results
        )
