import contextlib
import io
import os
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import anthropic
import openai
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem

import claude
import llm
import main
import openai_provider
import tools


def openai_reply(*items, status="completed"):
    return SimpleNamespace(
        output=list(items), output_text="Growing nicely.", status=status,
        error=None, incomplete_details=None,
    )


def function_call(call_id="call_1", name="list_folder", arguments="{}"):
    return ResponseFunctionToolCall(
        type="function_call", id="item_" + call_id, call_id=call_id,
        name=name, arguments=arguments,
    )


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.yolo = patch.object(tools, "yolo", False)
        self.yolo.start()
        self.addCleanup(self.yolo.stop)

    def provider(self, name, model=None):
        module, client_name = (
            (anthropic, "Anthropic") if name == "claude" else (openai, "OpenAI")
        )
        with patch.object(module, client_name):
            return llm.create_provider(name, model)

    def test_cli_defaults_and_overrides(self):
        args = main.parse_args([])
        self.assertEqual(args.provider, "claude")
        self.assertIsNone(args.model)
        self.assertFalse(args.yolo)
        args = main.parse_args(["--provider", "openai", "--model", "coding-model", "--yolo"])
        self.assertEqual((args.provider, args.model, args.yolo), ("openai", "coding-model", True))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main.parse_args(["--provider", "other"])

    def test_factory_and_models(self):
        for name, module in (("claude", claude), ("openai", openai_provider)):
            with self.subTest(provider=name):
                self.assertIsInstance(self.provider(name), module.Provider)
                self.assertEqual(self.provider(name).model, module.MODEL)
                self.assertEqual(self.provider(name, "custom").model, "custom")
        with self.assertRaises(ValueError):
            llm.create_provider("unknown")

    def test_only_selected_key_is_needed(self):
        for name, key in (("claude", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")):
            with self.subTest(provider=name), patch.dict(os.environ, {key: "test-placeholder"}, clear=True):
                provider = llm.create_provider(name)
                provider.client.close()

    def test_missing_keys_are_reported_without_traceback(self):
        with patch.dict(os.environ, {}, clear=True):
            for name in ("claude", "openai"):
                with self.subTest(provider=name), \
                        patch("builtins.input", side_effect=["hello", "quit"]), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    main.main(["--provider", name])
                    self.assertIn("Bonsai couldn't", output.getvalue())

    def test_claude_tool_round_trip_preserves_thinking_and_errors(self):
        provider = self.provider("claude", "custom-claude")
        thinking = SimpleNamespace(type="thinking", thinking="native reasoning")
        call = llm.ToolCall("claude_call", "list_folder", {})
        call.type = "tool_use"
        reply = SimpleNamespace(
            content=[thinking, SimpleNamespace(type="text", text="Looking."), call],
            stop_reason="tool_use",
        )
        final = SimpleNamespace(content=[], stop_reason="end_turn")
        provider.client.messages.create.side_effect = [reply, final]
        conversation = [{"role": "user", "content": "look around"}]
        with patch.dict(tools.FUNCTIONS, {"list_folder": Mock(side_effect=OSError("unavailable"))}):
            main.agent_loop(conversation, provider)
        self.assertIs(conversation[1]["content"], reply.content)
        self.assertIs(conversation[1]["content"][0], thinking)
        self.assertEqual(conversation[2], {
            "role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "claude_call",
                "content": "unavailable", "is_error": True,
            }],
        })
        request = provider.client.messages.create.call_args.kwargs
        self.assertEqual(request["model"], "custom-claude")
        self.assertEqual(request["tools"], tools.DEFINITIONS)
        self.assertIn("Bonsai", request["system"])

    def test_openai_multiple_calls_and_repeated_rounds(self):
        provider = self.provider("openai", "custom-openai")
        reasoning = ResponseReasoningItem(
            type="reasoning", id="reasoning_1", summary=[], encrypted_content="opaque",
        )
        replies = [
            openai_reply(reasoning, function_call(), function_call("call_2")),
            openai_reply(function_call("call_3")),
            openai_reply(),
        ]
        requests = []

        def respond(**kwargs):
            requests.append({**kwargs, "input": list(kwargs["input"])})
            return replies[len(requests) - 1]

        provider.client.responses.create.side_effect = respond
        conversation = [{"role": "user", "content": "look around"}]
        function = Mock(return_value="files")
        with patch.dict(tools.FUNCTIONS, {"list_folder": function}):
            main.agent_loop(conversation, provider)
        self.assertEqual(function.call_count, 3)
        function.assert_called_with()
        self.assertIs(conversation[1], reasoning)
        outputs = [item for item in conversation if isinstance(item, dict) and item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["call_1", "call_2", "call_3"])
        self.assertTrue(all(item["output"] == "files" for item in outputs))
        self.assertEqual(len(requests), 3)
        self.assertIn(reasoning, requests[1]["input"])
        self.assertEqual(requests[0]["model"], "custom-openai")
        self.assertFalse(requests[0]["store"])
        self.assertEqual(requests[0]["include"], ["reasoning.encrypted_content"])
        for definition, native in zip(tools.DEFINITIONS, requests[0]["tools"]):
            self.assertEqual(native["type"], "function")
            self.assertEqual(native["parameters"], definition["input_schema"])
            self.assertFalse(native["strict"])

    def test_invalid_arguments_and_unknown_tools_return_errors(self):
        provider = self.provider("openai")
        for arguments in ("{", "[]", "null", '"text"', "1"):
            with self.subTest(arguments=arguments):
                calls = provider.tool_calls_of(openai_reply(function_call(arguments=arguments)))
                function = Mock()
                with patch.dict(tools.FUNCTIONS, {"list_folder": function}):
                    results = tools.run(calls)
                function.assert_not_called()
                history = []
                provider.add_results(history, results)
                self.assertEqual(history[0]["call_id"], "call_1")
                self.assertTrue(history[0]["output"].startswith("Error: Invalid tool arguments:"))
        results = tools.run([llm.ToolCall("unknown", "unknown_tool", {})])
        self.assertTrue(results[0]["is_error"])

    def test_incomplete_openai_turn_is_not_executed(self):
        provider = self.provider("openai")
        for status in ("incomplete", "failed", "cancelled"):
            with self.subTest(status=status):
                provider.client.responses.create.return_value = openai_reply(function_call(), status=status)
                with patch.object(tools, "run") as run, self.assertRaises(llm.ResponseError):
                    main.agent_loop([], provider)
                run.assert_not_called()

    def test_api_failure_is_reported_and_executed_tools_are_not_replayed(self):
        for name in ("claude", "openai"):
            with self.subTest(provider=name):
                provider = self.provider(name)
                provider.Unreachable = llm.ResponseError
                if name == "openai":
                    provider.client.responses.create.side_effect = [
                        openai_reply(function_call()), llm.ResponseError("offline"), openai_reply(),
                    ]
                else:
                    call = SimpleNamespace(type="tool_use", id="call_1", name="list_folder", input={})
                    provider.client.messages.create.side_effect = [
                        SimpleNamespace(content=[call], stop_reason="tool_use"),
                        llm.ResponseError("offline"),
                        SimpleNamespace(content=[], stop_reason="end_turn"),
                    ]
                function = Mock(return_value="files")
                with patch.object(llm, "create_provider", return_value=provider), \
                        patch("builtins.input", side_effect=["look", "continue", "quit"]), \
                        patch.dict(tools.FUNCTIONS, {"list_folder": function}), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    main.main(["--provider", name])
                self.assertIn("couldn't complete", output.getvalue())
                function.assert_called_once()

    def test_yolo_prompt_and_main_flag(self):
        self.assertNotIn("Permissions are off", llm.system_prompt())
        self.assertIn("Before anything that can't be undone", llm.system_prompt(True))
        provider = self.provider("openai")
        with patch.object(llm, "create_provider", return_value=provider), \
                patch("builtins.input", side_effect=["quit"]):
            main.main(["--provider", "openai", "--yolo"])
        self.assertTrue(tools.yolo)


class ToolSafetyTests(unittest.TestCase):
    def test_declined_program_never_runs_and_yolo_skips_confirmation(self):
        call = llm.ToolCall("run", "run_program", {"program": "echo", "args": ["hello"]})
        function = Mock(return_value="hello")
        with patch.dict(tools.FUNCTIONS, {"run_program": function}), \
                patch.object(tools, "yolo", False), \
                patch("builtins.input", return_value="no"), \
                contextlib.redirect_stdout(io.StringIO()):
            results = tools.run([call])
            function.assert_not_called()
            self.assertTrue(results[0]["is_error"])
            self.assertIn("declined", results[0]["content"])
        with patch.dict(tools.FUNCTIONS, {"run_program": function}), \
                patch.object(tools, "yolo", True), \
                patch("builtins.input") as prompt, \
                contextlib.redirect_stdout(io.StringIO()):
            tools.run([call])
            prompt.assert_not_called()
            function.assert_called_once_with(program="echo", args=["hello"])

    def test_eof_declines_program(self):
        call = llm.ToolCall("run", "run_program", {"program": "echo"})
        with patch.object(tools, "yolo", False), patch("builtins.input", side_effect=EOFError):
            with self.assertRaises(PermissionError):
                tools.confirm(call)

    def test_timeout_and_output_truncation_are_preserved(self):
        with patch.object(tools.subprocess, "run", side_effect=subprocess.TimeoutExpired("echo", 2, output=b"partial")):
            with self.assertRaisesRegex(RuntimeError, "killed after 2 seconds.*\\npartial"):
                tools.run_program("echo", timeout_seconds=2)
        output = "a" * tools.MAX_OUTPUT + "ending"
        shortened = tools._shorten(output)
        self.assertTrue(shortened.startswith("a" * (tools.MAX_OUTPUT // 2)))
        self.assertTrue(shortened.endswith("ending"))
        self.assertIn("6 characters left out", shortened)


if __name__ == "__main__":
    unittest.main()
