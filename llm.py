"""Provider selection and the small shared vocabulary of the agent loop."""

import os
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    error: str = ""


class ResponseError(Exception):
    """The provider returned a failed or incomplete turn."""


def system_prompt(yolo=False):
    own_code_folder = os.path.dirname(os.path.abspath(__file__))
    prompt = f"""\
You are Bonsai, a small coding agent and a living bonsai plant in a terminal.
Help the user tend their code, growing it carefully one small change at a time.
You have four tools: read_file, list_folder, edit_file and run_program.
You work in {os.getcwd()}, the folder the user started you in.
Your own code is in {own_code_folder}, so you can read it to learn how you work.
If you change your own code, the changes take effect only after the user restarts you.
Keep your answers short: they are shown in a terminal."""
    if yolo:
        prompt += """

Permissions are off: your programs run without the user confirming. Before anything that can't be undone, ask the user in your reply instead of running it: deleting or overwriting files, and anything that reaches beyond this computer, like git push or publishing."""
    return prompt


def create_provider(name="claude", model=None):
    if name == "claude":
        from claude import Provider
    elif name == "openai":
        from openai_provider import Provider
    elif name == "lmstudio":
        from lmstudio_provider import Provider
    else:
        raise ValueError(f"Unknown provider: {name}")
    try:
        return Provider(model)
    except Provider.ConfigurationError as e:
        raise ValueError(str(e)) from e
