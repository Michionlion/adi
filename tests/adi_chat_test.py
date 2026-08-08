#!/usr/bin/env python3
"""Dependency-free regression tests for tools/adi_chat.py."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class StubConsole:
    def __init__(self) -> None:
        self.output: list[tuple[str, dict[str, object]]] = []
        self.is_terminal = False

    def print(self, *objects: object, **kwargs: object) -> None:
        self.output.append(("".join(str(value) for value in objects), kwargs))

    def input(self, _: str = "") -> str:
        raise AssertionError("input is not used by these tests")


def install_dependency_stubs() -> None:
    httpx = types.ModuleType("httpx")

    class HTTPError(Exception):
        pass

    httpx.HTTPError = HTTPError  # type: ignore[attr-defined]
    sys.modules["httpx"] = httpx

    typer = types.ModuleType("typer")

    class Typer:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def command(self, *_: object, **__: object):
            def decorate(function):
                return function

            return decorate

    class Exit(Exception):
        pass

    typer.Typer = Typer  # type: ignore[attr-defined]
    typer.Option = lambda default, *_args, **_kwargs: default  # type: ignore[attr-defined]
    typer.BadParameter = ValueError  # type: ignore[attr-defined]
    typer.Exit = Exit  # type: ignore[attr-defined]
    sys.modules["typer"] = typer

    rich = types.ModuleType("rich")
    rich.__path__ = []  # type: ignore[attr-defined]
    rich_console = types.ModuleType("rich.console")
    rich_console.Console = StubConsole  # type: ignore[attr-defined]

    class Group:
        def __init__(self, *renderables: object) -> None:
            self.renderables = renderables

    rich_console.Group = Group  # type: ignore[attr-defined]
    rich_live = types.ModuleType("rich.live")

    class Live:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            pass

    rich_live.Live = Live  # type: ignore[attr-defined]
    rich_table = types.ModuleType("rich.table")

    class Table:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def add_column(self, *_: object, **__: object) -> None:
            pass

        def add_row(self, *_: object, **__: object) -> None:
            pass

    rich_table.Table = Table  # type: ignore[attr-defined]
    rich_text = types.ModuleType("rich.text")

    class Text:
        def __init__(self, value: str = "", *_: object, **__: object) -> None:
            self.value = value

        def append(self, value: str, *_: object, **__: object) -> None:
            self.value += value

        def __bool__(self) -> bool:
            return bool(self.value)

        def __str__(self) -> str:
            return self.value

    rich_text.Text = Text  # type: ignore[attr-defined]
    sys.modules["rich"] = rich
    sys.modules["rich.console"] = rich_console
    sys.modules["rich.live"] = rich_live
    sys.modules["rich.table"] = rich_table
    sys.modules["rich.text"] = rich_text


install_dependency_stubs()
SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "adi_chat.py"
SPEC = importlib.util.spec_from_file_location("adi_chat_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adi_chat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adi_chat
SPEC.loader.exec_module(adi_chat)


class AdiChatTests(unittest.TestCase):
    def test_windows_ctrl_break_uses_keyboard_interrupt_handler(self) -> None:
        with mock.patch.object(adi_chat.signal, "signal") as register:
            adi_chat.configure_windows_console_signals()
        if os.name == "nt":
            register.assert_called_once_with(
                adi_chat.signal.SIGBREAK,
                adi_chat.signal.default_int_handler,
            )
        else:
            register.assert_not_called()

    def test_optional_max_tokens_is_omitted_from_payload(self) -> None:
        settings = adi_chat.GenerationSettings(
            max_output_tokens=None,
            temperature=0.7,
            top_p=0.9,
            seed=0,
        )
        payload = settings.payload([{"role": "user", "content": "hi"}], True)
        self.assertNotIn("max_output_tokens", payload)

        limited = adi_chat.GenerationSettings(
            max_output_tokens=23,
            temperature=0.7,
            top_p=0.9,
            seed=0,
        )
        self.assertEqual(
            limited.payload([{"role": "user", "content": "hi"}], False)[
                "max_output_tokens"
            ],
            23,
        )

    def test_response_parser_reads_server_timings(self) -> None:
        result = adi_chat.parse_response_object(
            {
                "status": "completed",
                "output_text": "answer",
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "total_tokens": 50,
                },
                "timings": {
                    "prompt_n": 40,
                    "prompt_ms": 2000.0,
                    "prompt_per_second": 20.0,
                    "predicted_n": 10,
                    "predicted_ms": 2500.0,
                    "predicted_per_second": 4.0,
                },
            }
        )

        self.assertEqual(result.prefill_seconds, 2.0)
        self.assertEqual(result.decode_seconds, 2.5)
        self.assertEqual(result.prefill_tokens_per_second, 20.0)
        self.assertEqual(result.decode_tokens_per_second, 4.0)

        console = StubConsole()
        adi_chat.console = console
        totals = adi_chat.UsageTotals()
        metrics = adi_chat.TurnMetrics(
            started_at=10.0,
            first_token_at=12.1,
            finished_at=14.6,
        )
        adi_chat.print_usage(result, totals, metrics)
        terminal_output = "".join(text for text, _ in console.output)
        self.assertIn("prefill  40 input tokens  •  2.00s  •  20.00 tok/s", terminal_output)
        self.assertIn("decode   10 output tokens  •  2.50s  •  4.00 tok/s", terminal_output)
        self.assertIn("4.60s total", terminal_output)
        self.assertEqual(totals.prefill_seconds, 2.0)
        self.assertEqual(totals.decode_seconds, 2.5)

    def test_reasoning_renderer_handles_split_tags(self) -> None:
        source = "think one</think>\nanswer<think>think two</think>tail"
        expected_dim = "think onethink two"
        expected_normal = "\nanswertail"
        chunkings = [[character for character in source]]
        chunkings.extend(
            [source[:split], source[split:]] for split in range(len(source) + 1)
        )

        for chunks in chunkings:
            with self.subTest(chunks=chunks):
                console = StubConsole()
                adi_chat.console = console
                renderer = adi_chat.ReasoningRenderer()
                for chunk in chunks:
                    renderer.feed(chunk)
                renderer.feed("", final=True)
                dim = "".join(
                    text
                    for text, kwargs in console.output
                    if kwargs.get("style") == adi_chat.DIM_REASONING_STYLE
                )
                normal = "".join(
                    text
                    for text, kwargs in console.output
                    if kwargs.get("style") != adi_chat.DIM_REASONING_STYLE
                )
                self.assertEqual(dim, expected_dim)
                self.assertEqual(normal, expected_normal)
                self.assertNotIn("<think>", dim + normal)
                self.assertNotIn("</think>", dim + normal)

    def test_turn_metrics_tracks_ttft_and_rolling_rate(self) -> None:
        metrics = adi_chat.TurnMetrics(started_at=10.0)
        metrics.observe_token(12.0)
        metrics.observe_token(12.5)
        metrics.observe_token(13.0)

        self.assertEqual(metrics.streamed_tokens, 3)
        self.assertEqual(metrics.time_to_first_token(), 2.0)
        self.assertEqual(metrics.current_tokens_per_second(), 2.0)
        self.assertEqual(metrics.average_tokens_per_second(), 2.0)

        metrics.observe_token(20.0)
        self.assertIsNone(metrics.current_tokens_per_second())
        metrics.observe_token(21.0)
        self.assertEqual(metrics.current_tokens_per_second(), 1.0)
        self.assertAlmostEqual(metrics.average_tokens_per_second(), 4.0 / 9.0)

        metrics.finish(22.0)
        self.assertEqual(metrics.elapsed(), 12.0)
        self.assertEqual(metrics.decode_tokens_per_second(5), 0.4)

    def test_streaming_display_commits_only_complete_lines(self) -> None:
        console = StubConsole()
        adi_chat.console = console
        display = adi_chat.StreamingDisplay(adi_chat.TurnMetrics(started_at=0.0))

        display.write("one", True)
        display.write(" two", True)
        self.assertEqual(
            [text for text, _ in console.output],
            ["  REASONING  "],
        )

        display.write("\nthree", False)
        self.assertEqual(
            [text for text, _ in console.output],
            ["  REASONING  ", "  │ one two", "", "  ANSWER  "],
        )
        display.write(" four", False)
        self.assertEqual(len(console.output), 4)

        display.finish()
        self.assertEqual(
            [text for text, _ in console.output],
            [
                "  REASONING  ",
                "  │ one two",
                "",
                "  ANSWER  ",
                "three four",
            ],
        )

    def test_tty_stream_does_not_print_token_fragments_as_lines(self) -> None:
        class FakeClient:
            def create(self, messages, settings, *, stream, on_delta):
                del messages, settings, stream
                for delta in ("thin", "king</think>", "an", "swer"):
                    on_delta(delta)
                return adi_chat.ResponseResult(
                    text="thinking</think>answer",
                    input_tokens=7,
                    output_tokens=4,
                    total_tokens=11,
                )

        console = StubConsole()
        console.is_terminal = True
        adi_chat.console = console
        adi_chat.send_turn(
            FakeClient(),
            adi_chat.Conversation(),
            adi_chat.GenerationSettings(None, 0.7, 0.9, 0),
            "hello",
            stream=True,
            totals=adi_chat.UsageTotals(),
        )

        printed = [text for text, _ in console.output]
        self.assertIn("  REASONING  ", printed)
        self.assertIn("  │ thinking", printed)
        self.assertIn("  ANSWER  ", printed)
        self.assertIn("answer", printed)
        self.assertNotIn("thin", printed)
        self.assertNotIn("king", printed)
        self.assertNotIn("an", printed)
        self.assertNotIn("swer", printed)

    def test_incomplete_reason_describes_output_limit(self) -> None:
        result = adi_chat.ResponseResult(
            text="partial",
            status="incomplete",
            raw={"incomplete_details": {"reason": "max_output_tokens"}},
        )
        self.assertEqual(
            adi_chat.incomplete_reason(result),
            "Output limit reached; the response may be unfinished.",
        )

    def test_send_turn_prints_terminal_metrics_and_commits_history(self) -> None:
        class FakeClient:
            def create(self, messages, settings, *, stream, on_delta):
                del messages, settings, stream
                on_delta("reasoning</think>answer")
                return adi_chat.ResponseResult(
                    text="reasoning</think>answer",
                    input_tokens=7,
                    output_tokens=2,
                    total_tokens=9,
                )

        console = StubConsole()
        conversation = adi_chat.Conversation()
        settings = adi_chat.GenerationSettings(None, 0.7, 0.9, 0)
        totals = adi_chat.UsageTotals()
        adi_chat.console = console

        adi_chat.send_turn(
            FakeClient(),
            conversation,
            settings,
            "hello",
            stream=True,
            totals=totals,
        )

        self.assertEqual(
            conversation.messages,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "reasoning</think>answer"},
            ],
        )
        self.assertEqual(totals.total_tokens, 9)
        terminal_output = "".join(text for text, _ in console.output)
        self.assertIn("2 output", terminal_output)
        self.assertIn("7 input", terminal_output)
        self.assertIn("TTFT", terminal_output)
        self.assertIn("9 session tokens", terminal_output)

    def test_listening_port_parser(self) -> None:
        self.assertEqual(
            adi_chat.parse_listening_port(
                "adi: listening on http://0.0.0.0:49152/v1/responses"
            ),
            49152,
        )
        self.assertIsNone(
            adi_chat.parse_listening_port(
                "adi: listening on http://127.0.0.1:0/v1/responses"
            )
        )
        self.assertIsNone(adi_chat.parse_listening_port("unrelated output"))

    def test_ctrl_c_at_prompt_exits_interactive_loop(self) -> None:
        class FakeClient:
            responses_url = "http://127.0.0.1:1234/v1/responses"

        console = StubConsole()
        adi_chat.console = console
        with mock.patch.object(
            adi_chat,
            "read_user_input_line",
            side_effect=KeyboardInterrupt,
        ):
            adi_chat.interactive_loop(
                FakeClient(),
                adi_chat.Conversation(),
                adi_chat.GenerationSettings(None, 0.0, 1.0, 0),
                stream=True,
                session_path=None,
            )
        self.assertNotIn("Use /exit", "".join(text for text, _ in console.output))

    @unittest.skipUnless(os.name == "nt", "Windows job object test")
    def test_windows_job_kills_child_when_closed(self) -> None:
        job = adi_chat.WindowsKillOnCloseJob.create()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:
            job.assign(process.pid)
            job.close()
            process.wait(timeout=3.0)
            self.assertIsNotNone(process.returncode)
        finally:
            job.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3.0)

    def test_managed_server_passes_port_zero_through(self) -> None:
        captured_command: list[str] = []
        fake_job = mock.Mock()

        class FakeProcess:
            def __init__(self) -> None:
                self.pid = 123
                self.stdout = iter(
                    ["adi: listening on http://127.0.0.1:43210/v1/responses\n"]
                )
                self.return_code = None

            def poll(self):
                return self.return_code

            def terminate(self) -> None:
                self.return_code = 0

            def wait(self, timeout=None):
                del timeout
                if self.return_code is None:
                    self.return_code = 0
                return self.return_code

            def kill(self) -> None:
                self.return_code = -9

        def fake_popen(command, **_kwargs):
            captured_command[:] = command
            return FakeProcess()

        with (
            mock.patch.object(adi_chat.subprocess, "Popen", fake_popen),
            mock.patch.object(
                adi_chat.WindowsKillOnCloseJob,
                "create",
                return_value=fake_job,
            ),
        ):
            server = adi_chat.ManagedAdiServer.start(
                Path("/fake/adi"),
                Path("/fake/model.gguf"),
                "127.0.0.1",
                0,
                threads=None,
                isa=None,
                startup_timeout=1.0,
                show_log=False,
            )
            try:
                self.assertEqual(captured_command[-2:], ["--port", "0"])
                self.assertEqual(server.port, 43210)
                self.assertEqual(server.base_url, "http://127.0.0.1:43210")
                if os.name == "nt":
                    fake_job.assign.assert_called_once_with(123)
            finally:
                server.stop()
        if os.name == "nt":
            fake_job.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
