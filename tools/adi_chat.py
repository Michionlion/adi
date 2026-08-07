#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27,<1",
#   "rich>=13.7,<15",
#   "typer>=0.12,<1",
# ]
# ///
"""Start or connect to ADI and provide a small interactive chat harness."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import httpx
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError as error:  # pragma: no cover - exercised only before install
    missing = getattr(error, "name", "a required package")
    print(
        f"Missing {missing}. Run with `uv run tools/adi_chat.py ...` or install "
        "`httpx typer rich`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


app = typer.Typer(
    add_completion=False,
    help=(
        "Start ADI (or connect to an existing ADI server) and chat through its "
        "Responses API."
    ),
    no_args_is_help=True,
)
console = Console()


class ClientError(RuntimeError):
    """A concise, user-facing client failure."""


class ApiError(ClientError):
    """An error returned by ADI's Responses API."""


@dataclass(frozen=True)
class GenerationSettings:
    max_output_tokens: int
    temperature: float
    top_p: float
    seed: int
    model_id: Optional[str] = None

    def payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": messages,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": stream,
        }
        if self.model_id:
            payload["model"] = self.model_id
        return payload


@dataclass
class ResponseResult:
    text: str
    status: str = "completed"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, result: ResponseResult) -> None:
        self.requests += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.total_tokens += result.total_tokens


@dataclass
class Conversation:
    system_prompt: Optional[str] = None
    messages: list[dict[str, str]] = field(default_factory=list)

    def request_messages(self, user_text: str) -> list[dict[str, str]]:
        request: list[dict[str, str]] = []
        if self.system_prompt:
            request.append({"role": "system", "content": self.system_prompt})
        request.extend(self.messages)
        request.append({"role": "user", "content": user_text})
        return request

    def commit(self, user_text: str, assistant_text: str) -> None:
        self.messages.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        )

    def reset(self) -> None:
        self.messages.clear()

    def save(self, path: Path) -> None:
        path = path.expanduser()
        document = {
            "version": 1,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ClientError(f"cannot save session {path}: {error}") from error

    @classmethod
    def load(cls, path: Path) -> "Conversation":
        path = path.expanduser()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ClientError(f"cannot load session {path}: {error}") from error
        if not isinstance(document, dict) or document.get("version") != 1:
            raise ClientError(f"unsupported session format in {path}")
        system_prompt = document.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ClientError(f"invalid system prompt in {path}")
        messages = document.get("messages")
        if not isinstance(messages, list):
            raise ClientError(f"invalid message list in {path}")
        validated: list[dict[str, str]] = []
        expected_role = "user"
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ClientError(f"invalid message {index} in {path}")
            role = message.get("role")
            content = message.get("content")
            if role != expected_role or not isinstance(content, str):
                raise ClientError(
                    f"session messages must alternate user/assistant at message {index}"
                )
            validated.append({"role": role, "content": content})
            expected_role = "assistant" if role == "user" else "user"
        if expected_role == "assistant":
            raise ClientError(f"session {path} ends with an unanswered user message")
        return cls(system_prompt=system_prompt, messages=validated)


class AdiResponsesClient:
    def __init__(self, base_url: str) -> None:
        self.responses_url = responses_url(base_url)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
            trust_env=False,
            headers={"Accept": "application/json, text/event-stream"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AdiResponsesClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def create(
        self,
        messages: list[dict[str, str]],
        settings: GenerationSettings,
        *,
        stream: bool,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> ResponseResult:
        payload = settings.payload(messages, stream)
        if stream:
            return self._create_stream(payload, on_delta or (lambda _: None))
        return self._create_nonstream(payload)

    def _create_nonstream(self, payload: dict[str, Any]) -> ResponseResult:
        try:
            response = self._client.post(self.responses_url, json=payload)
        except httpx.HTTPError as error:
            raise ClientError(f"request to ADI failed: {error}") from error
        self._raise_for_error(response)
        try:
            body = response.json()
        except json.JSONDecodeError as error:
            raise ApiError("ADI returned invalid JSON") from error
        if not isinstance(body, dict):
            raise ApiError("ADI returned a non-object response")
        return parse_response_object(body)

    def _create_stream(
        self,
        payload: dict[str, Any],
        on_delta: Callable[[str], None],
    ) -> ResponseResult:
        chunks: list[str] = []
        terminal_response: Optional[dict[str, Any]] = None
        try:
            with self._client.stream(
                "POST",
                self.responses_url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                self._raise_for_error(response)
                event_name = ""
                data_lines: list[str] = []

                def dispatch() -> None:
                    nonlocal terminal_response, event_name, data_lines
                    if not data_lines:
                        event_name = ""
                        return
                    raw_data = "\n".join(data_lines)
                    event_name = event_name.strip()
                    data_lines = []
                    if raw_data == "[DONE]":
                        event_name = ""
                        return
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError as error:
                        raise ApiError(f"ADI returned invalid SSE JSON: {raw_data!r}") from error
                    if not isinstance(event, dict):
                        raise ApiError("ADI returned a non-object SSE event")
                    event_type = event.get("type") or event_name
                    event_name = ""
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise ApiError("ADI returned an invalid text delta")
                        chunks.append(delta)
                        on_delta(delta)
                        return
                    if event_type in {"response.completed", "response.incomplete"}:
                        response_object = event.get("response")
                        if not isinstance(response_object, dict):
                            raise ApiError("ADI terminal event omitted its response object")
                        terminal_response = response_object
                        return
                    if event_type == "error":
                        message = event.get("message")
                        if not isinstance(message, str):
                            nested = event.get("error")
                            message = (
                                nested.get("message")
                                if isinstance(nested, dict)
                                else "ADI streaming request failed"
                            )
                        raise ApiError(str(message))

                for line in response.iter_lines():
                    if line == "":
                        dispatch()
                    elif line.startswith(":"):
                        continue
                    elif line.startswith("event:"):
                        event_name = line[6:].lstrip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                dispatch()
        except ApiError:
            raise
        except httpx.HTTPError as error:
            raise ClientError(f"stream from ADI failed: {error}") from error

        if terminal_response is not None:
            result = parse_response_object(terminal_response)
            if not result.text and chunks:
                result.text = "".join(chunks)
            return result
        raise ApiError("ADI stream ended before a terminal response")

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        try:
            body = response.read().decode(response.encoding or "utf-8", errors="replace")
        except (httpx.HTTPError, UnicodeError):
            body = ""
        message = f"ADI returned HTTP {response.status_code}"
        if body:
            try:
                document = json.loads(body)
                if isinstance(document, dict):
                    error = document.get("error")
                    if isinstance(error, dict) and isinstance(error.get("message"), str):
                        message = error["message"]
                    elif isinstance(document.get("message"), str):
                        message = document["message"]
            except json.JSONDecodeError:
                message = f"{message}: {body[:300]}"
        raise ApiError(message)


class ManagedAdiServer:
    def __init__(
        self,
        process: subprocess.Popen[str],
        log_tail: deque[str],
        log_thread: threading.Thread,
    ) -> None:
        self.process = process
        self.log_tail = log_tail
        self.log_thread = log_thread

    @classmethod
    def start(
        cls,
        executable: Path,
        model: Path,
        host: str,
        port: int,
        *,
        threads: Optional[int],
        isa: Optional[str],
        startup_timeout: float,
        show_log: bool,
    ) -> "ManagedAdiServer":
        if port_is_open(host_for_connect(host), port):
            raise ClientError(
                f"{host}:{port} is already accepting connections; use --connect "
                "or choose another --port"
            )
        environment = os.environ.copy()
        if threads is not None:
            environment["ADI_THREADS"] = str(threads)
        if isa is not None:
            environment["ADI_CPU_ISA"] = isa
        command = [
            str(executable),
            "serve",
            "--model",
            str(model),
            "--host",
            host,
            "--port",
            str(port),
        ]
        creation_flags = 0
        start_new_session = False
        if os.name == "nt":
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
                start_new_session=start_new_session,
            )
        except OSError as error:
            raise ClientError(f"cannot start ADI: {error}") from error

        log_tail: deque[str] = deque(maxlen=40)

        def drain_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stripped = line.rstrip("\r\n")
                log_tail.append(stripped)
                if show_log:
                    console.print(f"[adi] {stripped}", style="dim", markup=False)

        log_thread = threading.Thread(
            target=drain_output,
            name="adi-log-reader",
            daemon=True,
        )
        log_thread.start()
        server = cls(process, log_tail, log_thread)
        try:
            server.wait_until_ready(host, port, startup_timeout)
        except BaseException:
            server.stop()
            raise
        return server

    def wait_until_ready(self, host: str, port: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        connect_host = host_for_connect(host)
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                tail = "\n".join(self.log_tail)
                detail = f"\n{tail}" if tail else ""
                raise ClientError(f"ADI exited with status {return_code}{detail}")
            if port_is_open(connect_host, port):
                return
            time.sleep(0.1)
        tail = "\n".join(self.log_tail)
        detail = f"\nRecent ADI output:\n{tail}" if tail else ""
        raise ClientError(f"ADI did not listen within {timeout:g} seconds{detail}")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        self.log_thread.join(timeout=1.0)

    def __enter__(self) -> "ManagedAdiServer":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def responses_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ClientError("--connect must start with http:// or https://")
    if base.endswith("/v1"):
        return base + "/responses"
    if base.endswith("/v1/responses"):
        return base
    return base + "/v1/responses"


def parse_response_object(body: dict[str, Any]) -> ResponseResult:
    output = body.get("output")
    text_parts: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    text_parts.append(part["text"])
    if not text_parts and isinstance(body.get("output_text"), str):
        text_parts.append(body["output_text"])
    usage = body.get("usage")
    input_tokens = output_tokens = total_tokens = 0
    if isinstance(usage, dict):
        input_tokens = integer_or_zero(usage.get("input_tokens"))
        output_tokens = integer_or_zero(usage.get("output_tokens"))
        total_tokens = integer_or_zero(usage.get("total_tokens"))
    return ResponseResult(
        text="".join(text_parts),
        status=str(body.get("status", "completed")),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
        raw=body,
    )


def integer_or_zero(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def host_for_connect(host: str) -> str:
    return "127.0.0.1" if host == "0.0.0.0" else host


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def choose_free_port(host: str) -> int:
    bind_host = host_for_connect(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((bind_host, 0))
        return int(listener.getsockname()[1])


def resolve_adi_executable(requested: Optional[str]) -> Path:
    candidates: list[Path] = []
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.parent != Path(".") or requested_path.is_absolute():
            candidates.append(requested_path)
        else:
            resolved = shutil.which(requested)
            if resolved:
                candidates.append(Path(resolved))
            candidates.append(requested_path)
    environment_path = os.environ.get("ADI_EXE")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    for name in ("adi.exe", "adi"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))

    roots = [Path.cwd()]
    try:
        roots.append(Path(__file__).resolve().parents[1])
    except IndexError:
        pass
    relative_candidates = (
        Path("build/Release/adi.exe"),
        Path("build/RelWithDebInfo/adi.exe"),
        Path("build/Debug/adi.exe"),
        Path("build/adi"),
        Path("adi.exe"),
        Path("adi"),
    )
    for root in roots:
        candidates.extend(root / relative for relative in relative_candidates)

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser().resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file() and (os.name == "nt" or os.access(normalized, os.X_OK)):
            return normalized
    searched = "\n  ".join(str(path) for path in seen)
    raise ClientError(
        "cannot find the ADI executable; pass --adi or set ADI_EXE. Searched:\n  "
        + searched
    )


def print_usage(result: ResponseResult, totals: UsageTotals) -> None:
    totals.add(result)
    console.print(
        "[dim]"
        f"request: {result.input_tokens} input + {result.output_tokens} output "
        f"= {result.total_tokens} tokens; session: {totals.total_tokens} tokens"
        "[/dim]"
    )


def print_history(conversation: Conversation) -> None:
    table = Table(title="Conversation history", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Role", style="bold")
    table.add_column("Content")
    if conversation.system_prompt:
        table.add_row("0", "system", truncate(conversation.system_prompt))
    for index, message in enumerate(conversation.messages, start=1):
        table.add_row(str(index), message["role"], truncate(message["content"]))
    console.print(table)


def truncate(value: str, limit: int = 120) -> str:
    flattened = " ".join(value.split())
    return flattened if len(flattened) <= limit else flattened[: limit - 1] + "…"


def strip_outer_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def read_pasted_message() -> str:
    console.print("[dim]Paste text; enter a line containing only `.` to send.[/dim]")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()



def save_automatic_session(
    conversation: Conversation,
    session_path: Optional[Path],
) -> None:
    if session_path is None:
        return
    try:
        conversation.save(session_path)
    except ClientError as error:
        console.print(str(error), style="red", markup=False)


def print_repl_help() -> None:
    console.print(
        "[bold]Commands[/bold]\n"
        "  /help                 Show this help\n"
        "  /exit, /quit          Exit\n"
        "  /reset                Clear user/assistant history\n"
        "  /system [TEXT]        Show or replace the system prompt\n"
        "  /history              Show abbreviated conversation history\n"
        "  /save PATH            Save the conversation as JSON\n"
        "  /load PATH            Load a saved conversation\n"
        "  /paste                Enter a multi-line message\n"
        "  /stats                Show cumulative token usage"
    )


def send_turn(
    client: AdiResponsesClient,
    conversation: Conversation,
    settings: GenerationSettings,
    user_text: str,
    *,
    stream: bool,
    totals: UsageTotals,
) -> ResponseResult:
    console.print("[bold cyan]assistant[/bold cyan]")
    printed_delta = False

    def on_delta(delta: str) -> None:
        nonlocal printed_delta
        printed_delta = printed_delta or bool(delta)
        console.print(delta, end="", markup=False, highlight=False, soft_wrap=True)

    try:
        result = client.create(
            conversation.request_messages(user_text),
            settings,
            stream=stream,
            on_delta=on_delta,
        )
    except KeyboardInterrupt:
        if printed_delta:
            console.print()
        console.print("[yellow]Generation interrupted; the turn was not saved.[/yellow]")
        raise
    if stream:
        console.print()
    else:
        console.print(result.text, markup=False, highlight=False, soft_wrap=True)
    if not result.text:
        console.print("[yellow]ADI returned no output text.[/yellow]")
    conversation.commit(user_text, result.text)
    print_usage(result, totals)
    return result


def interactive_loop(
    client: AdiResponsesClient,
    conversation: Conversation,
    settings: GenerationSettings,
    *,
    stream: bool,
    session_path: Optional[Path],
) -> None:
    totals = UsageTotals()
    console.print(
        "[bold green]ADI chat ready.[/bold green] "
        "Type /help for commands; Ctrl+D or /exit quits."
    )
    while True:
        try:
            line = console.input("[bold green]you[/bold green] > ").strip()
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            console.print("\n[dim]Use /exit to quit.[/dim]")
            continue
        if not line:
            continue
        if line.startswith("/"):
            command, _, argument = line.partition(" ")
            command = command.lower()
            argument = argument.strip()
            if command in {"/exit", "/quit"}:
                return
            if command == "/help":
                print_repl_help()
                continue
            if command == "/reset":
                conversation.reset()
                save_automatic_session(conversation, session_path)
                console.print("[dim]Conversation history cleared.[/dim]")
                continue
            if command == "/system":
                if argument:
                    conversation.system_prompt = argument
                    save_automatic_session(conversation, session_path)
                    console.print("[dim]System prompt updated.[/dim]")
                elif conversation.system_prompt:
                    console.print(
                        conversation.system_prompt,
                        markup=False,
                        highlight=False,
                    )
                else:
                    console.print("[dim](none)[/dim]")
                continue
            if command == "/history":
                print_history(conversation)
                continue
            if command == "/stats":
                console.print(
                    f"[dim]{totals.requests} requests, {totals.input_tokens} input, "
                    f"{totals.output_tokens} output, {totals.total_tokens} total tokens[/dim]"
                )
                continue
            if command in {"/save", "/load"}:
                if not argument:
                    console.print(f"{command} requires a path.", style="yellow", markup=False)
                    continue
                path = Path(strip_outer_quotes(argument)).expanduser()
                try:
                    if command == "/save":
                        conversation.save(path)
                        console.print(f"Saved {path}.", style="dim", markup=False)
                    else:
                        loaded = Conversation.load(path)
                        conversation.system_prompt = loaded.system_prompt
                        conversation.messages = loaded.messages
                        console.print(f"Loaded {path}.", style="dim", markup=False)
                except ClientError as error:
                    console.print(str(error), style="red", markup=False)
                continue
            if command == "/paste":
                try:
                    line = read_pasted_message()
                except KeyboardInterrupt:
                    console.print("\n[dim]Paste cancelled.[/dim]")
                    continue
                if not line:
                    continue
            else:
                console.print(f"Unknown command: {command}", style="yellow", markup=False)
                continue
        try:
            send_turn(
                client,
                conversation,
                settings,
                line,
                stream=stream,
                totals=totals,
            )
        except KeyboardInterrupt:
            continue
        except ClientError as error:
            console.print(str(error), style="red", markup=False)
            continue
        save_automatic_session(conversation, session_path)


@app.command()
def chat(
    model: Optional[Path] = typer.Option(
        None,
        "--model",
        help="Mach GGUF to serve. Required unless --connect is used.",
        dir_okay=False,
        resolve_path=True,
    ),
    adi: Optional[str] = typer.Option(
        None,
        "--adi",
        help="ADI executable or command name. Auto-detected when omitted.",
    ),
    connect: Optional[str] = typer.Option(
        None,
        "--connect",
        metavar="URL",
        help="Connect to an already-running ADI server instead of starting one.",
    ),
    host: str = typer.Option("127.0.0.1", help="Host passed to `adi serve`."),
    port: int = typer.Option(
        8080,
        min=0,
        max=65535,
        help="Port passed to ADI. Use 0 to choose a free local port.",
    ),
    threads: Optional[int] = typer.Option(
        None,
        min=1,
        help="Set ADI_THREADS for the child process.",
    ),
    isa: Optional[str] = typer.Option(
        None,
        help="Set ADI_CPU_ISA (scalar, avx2, avx512, neon, or sve).",
    ),
    startup_timeout: float = typer.Option(
        600.0,
        min=0.1,
        help="Seconds to wait for model loading and the listening socket.",
    ),
    server_log: bool = typer.Option(
        False,
        "--server-log/--no-server-log",
        help="Mirror ADI stdout/stderr into the chat terminal.",
    ),
    model_id: Optional[str] = typer.Option(
        None,
        help="Optional Responses API model name; omitted by default.",
    ),
    system: Optional[str] = typer.Option(
        None,
        help="System prompt placed before the conversation history.",
    ),
    session: Optional[Path] = typer.Option(
        None,
        help="Load and continuously save conversation state to this JSON file.",
        dir_okay=False,
        resolve_path=True,
    ),
    prompt: Optional[str] = typer.Option(
        None,
        help="Send one prompt and exit instead of entering the REPL.",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        min=1,
        help="Maximum output tokens per turn.",
    ),
    temperature: float = typer.Option(
        0.7,
        min=0.0,
        max=2.0,
        help="Sampling temperature; 0 enables greedy decoding.",
    ),
    top_p: float = typer.Option(
        0.9,
        min=0.000001,
        max=1.0,
        help="Nucleus-sampling probability mass.",
    ),
    seed: int = typer.Option(0, min=0, help="Deterministic sampling seed."),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Use Responses API server-sent-event streaming.",
    ),
) -> None:
    """Run the ADI chat client."""
    if temperature != 0.0 and temperature < 0.0001:
        raise typer.BadParameter("temperature must be 0 or at least 0.0001")
    if isa not in {None, "scalar", "avx2", "avx512", "neon", "sve"}:
        raise typer.BadParameter("isa must be scalar, avx2, avx512, neon, or sve")
    managed_server: Optional[ManagedAdiServer] = None
    try:
        if session and session.exists():
            conversation = Conversation.load(session)
        else:
            conversation = Conversation()
        if system is not None:
            conversation.system_prompt = system

        settings = GenerationSettings(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            model_id=model_id,
        )

        if connect:
            base_url = connect
        else:
            if model is None:
                raise ClientError("--model is required unless --connect is used")
            model = model.expanduser().resolve()
            if not model.is_file():
                raise ClientError(f"model does not exist: {model}")
            executable = resolve_adi_executable(adi)
            selected_port = choose_free_port(host) if port == 0 else port
            console.print(
                f"[dim]Starting {executable} on {host}:{selected_port}; "
                "model loading may take a while.[/dim]"
            )
            managed_server = ManagedAdiServer.start(
                executable,
                model,
                host,
                selected_port,
                threads=threads,
                isa=isa,
                startup_timeout=startup_timeout,
                show_log=server_log,
            )
            base_url = f"http://{host_for_connect(host)}:{selected_port}"
            console.print(f"ADI is listening at {base_url}.", style="dim", markup=False)

        with AdiResponsesClient(base_url) as client:
            if prompt is not None:
                totals = UsageTotals()
                try:
                    send_turn(
                        client,
                        conversation,
                        settings,
                        prompt,
                        stream=stream,
                        totals=totals,
                    )
                except KeyboardInterrupt as error:
                    raise typer.Exit(130) from error
                if session:
                    conversation.save(session)
            else:
                interactive_loop(
                    client,
                    conversation,
                    settings,
                    stream=stream,
                    session_path=session,
                )
    except ClientError as error:
        console.print(f"error: {error}", style="red", markup=False)
        raise typer.Exit(1) from error
    finally:
        if managed_server is not None:
            console.print("[dim]Stopping ADI.[/dim]")
            managed_server.stop()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()
