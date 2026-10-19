"""Bonsai: a small coding agent grown from the Sancho example."""

import argparse
import readline  # importing it is enough: input() gets line editing and history

from rich.console import Console
from rich.markdown import Markdown

import llm
import tools

console = Console()


def agent_loop(conversation, provider):
    """Keep asking the model and running its tools until the turn is done."""
    while True:
        message = provider.get_next_answer(conversation, tools.DEFINITIONS, yolo=tools.yolo)
        provider.record_answer(conversation, message)

        text = provider.text_of(message)
        if text:
            console.print("Bonsai:", Markdown(text))

        tool_calls = provider.tool_calls_of(message)
        if not tool_calls:
            return

        results = tools.run(tool_calls)
        provider.add_results(conversation, results)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bonsai, a small coding agent 🌱")
    parser.add_argument("--provider", choices=("claude", "openai"), default="claude")
    parser.add_argument("--model", help="model override for the selected provider")
    parser.add_argument("--yolo", action="store_true", help="run programs without confirmation")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        provider = llm.create_provider(args.provider, args.model)
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"Bonsai couldn't initialize {args.provider}: {e}")
        return
    conversation = []
    tools.yolo = args.yolo
    if tools.yolo:
        print(
            "\033[33mWarning: --yolo is on. Bonsai will run programs without asking.\033[0m"
        )
    print(f"Bonsai 🌱  Provider: {args.provider}  Model: {provider.model}")
    print("Let's grow some code. (adiós or Ctrl-D to quit)")

    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.strip().lower() in ["quit", "exit", "adios", "adiós"]:
                break

            if not user_input.strip():
                continue

            conversation.append({"role": "user", "content": user_input})
            try:
                agent_loop(conversation, provider)
            except (provider.Unreachable, llm.ResponseError) as e:
                print(
                    f"Bonsai couldn't complete the {args.provider} turn: {e} Try again in a moment."
                )
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
