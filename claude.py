"""A thin shim over the Anthropic SDK, with names we find clearer."""

import anthropic

from llm import ResponseError, system_prompt

MODEL = "claude-sonnet-5-5"


class Provider:
    Unreachable = anthropic.APIError
    ConfigurationError = anthropic.AnthropicError

    def __init__(self, model=None):
        self.model = model or MODEL
        self.client = anthropic.Anthropic()

    def get_next_answer(self, conversation, tools, yolo=False):
        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system_prompt(yolo),
                messages=conversation,
                tools=tools,
            )
        except TypeError as e:
            # Some SDK versions defer missing-credential validation to the request.
            raise ResponseError(str(e)) from e

    def record_answer(self, conversation, message):
        # Keep thinking blocks unchanged alongside text and tool calls.
        conversation.append({"role": "assistant", "content": message.content})

    def text_of(self, message):
        return "".join(block.text for block in message.content if block.type == "text")

    def tool_calls_of(self, message):
        if message.stop_reason != "tool_use":
            return []
        return [block for block in message.content if block.type == "tool_use"]

    def add_results(self, conversation, results):
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result["id"],
                    "content": result["content"],
                    **({"is_error": True} if result.get("is_error") else {}),
                }
                for result in results
            ],
        })
