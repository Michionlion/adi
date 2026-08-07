#!/usr/bin/env python3
"""Dependency-free regression tests for tools/adi_chat.py."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class StubConsole:
    def __init__(self) -> None:
        self.output: list[tuple[str, dict[str, object]]] = []

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
    rich_table = types.ModuleType("rich.table")

    class Table:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def add_column(self, *_: object, **__: object) -> None:
            pass

        def add_row(self, *_: object, **__: object) -> None:
            pass

    rich_table.Table = Table  # type: ignore[attr-defined]
    sys.modules["rich"] = rich
    sys.modules["rich.console"] = rich_console
    sys.modules["rich.table"] = rich_table


install_dependency_stubs()
SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "adi_chat.py"
SPEC = importlib.util.spec_from_file_location("adi_chat_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
adi_chat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adi_chat
SPEC.loader.exec_module(adi_chat)


class AdiChatTests(unittest.TestCase):
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

    def test_managed_server_passes_port_zero_through(self) -> None:
        captured_command: list[str] = []

        class FakeProcess:
            def __init__(self) -> None:
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

        with mock.patch.object(adi_chat.subprocess, "Popen", fake_popen):
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
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
