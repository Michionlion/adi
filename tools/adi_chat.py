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
import math
import os
import queue
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import httpx
    import typer
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
except ImportError as error:  # pragma: no cover - exercised only before install
    missing = getattr(error, "name", "a required package")
    print(
        f"Missing {missing}. Run with `uv run tools/adi_chat.py ...` or install "
        "`httpx typer rich`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error

try:
    import readline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for better editing
    readline = None  # type: ignore[assignment]


app = typer.Typer(
    add_completion=False,
    help=(
        "Start ADI (or connect to an existing ADI server) and chat through its "
        "Responses API."
    ),
    no_args_is_help=True,
)
console = Console()

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
THINK_SUFFIX_GUARD = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG)) - 1
DIM_REASONING_STYLE = "dim italic"
REASONING_GUTTER_STYLE = "bold cyan"
REASONING_LABEL_STYLE = "bold white on #374151"
ANSWER_LABEL_STYLE = "bold white on #0f766e"
STATUS_BAR_STYLE = "bold white on #334155"
READLINE_PROMPT = "\x01\x1b[32m\x02user\x01\x1b[0m\x02 > "
RATE_WINDOW_SECONDS = 5.0


class ClientError(RuntimeError):
    """A concise, user-facing client failure."""


class ApiError(ClientError):
    """An error returned by ADI's Responses API."""


def configure_windows_utf8_stdio() -> None:
    """Keep redirected Windows output compatible with UTF-8 PowerShell pipes."""
    if os.name != "nt":
        return
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def configure_windows_console_signals() -> None:
    """Treat Windows Ctrl+Break like Ctrl+C inside the client."""
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)


@dataclass(frozen=True)
class GenerationSettings:
    max_output_tokens: Optional[int]
    temperature: float
    top_p: float
    seed: int
    model_id: Optional[str] = None

    def payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": stream,
        }
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
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
    prefill_seconds: Optional[float] = None
    decode_seconds: Optional[float] = None
    prefill_tokens_per_second: Optional[float] = None
    decode_tokens_per_second: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnMetrics:
    """Client-observed timings for one Responses API request."""

    started_at: float = field(default_factory=time.monotonic)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    streamed_tokens: int = 0
    recent_token_times: deque[float] = field(default_factory=deque)

    def observe_token(self, observed_at: Optional[float] = None) -> None:
        observed_at = time.monotonic() if observed_at is None else observed_at
        if self.first_token_at is None:
            self.first_token_at = observed_at
        self.streamed_tokens += 1
        self.recent_token_times.append(observed_at)
        cutoff = observed_at - RATE_WINDOW_SECONDS
        while self.recent_token_times[0] < cutoff:
            self.recent_token_times.popleft()

    def finish(self, finished_at: Optional[float] = None) -> None:
        self.finished_at = time.monotonic() if finished_at is None else finished_at

    def elapsed(self, now: Optional[float] = None) -> float:
        end = self.finished_at
        if end is None:
            end = time.monotonic() if now is None else now
        return max(0.0, end - self.started_at)

    def time_to_first_token(self) -> Optional[float]:
        if self.first_token_at is None:
            return None
        return max(0.0, self.first_token_at - self.started_at)

    def current_tokens_per_second(self) -> Optional[float]:
        if len(self.recent_token_times) < 2:
            return None
        duration = self.recent_token_times[-1] - self.recent_token_times[0]
        if duration <= 0.0:
            return None
        return (len(self.recent_token_times) - 1) / duration

    def average_tokens_per_second(self) -> Optional[float]:
        if self.streamed_tokens < 2 or self.first_token_at is None:
            return None
        last_token_at = self.recent_token_times[-1]
        duration = last_token_at - self.first_token_at
        if duration <= 0.0:
            return None
        return (self.streamed_tokens - 1) / duration

    def decode_tokens_per_second(self, output_tokens: int) -> Optional[float]:
        if output_tokens < 2:
            return None
        if self.first_token_at is None or self.finished_at is None:
            return None
        duration = self.finished_at - self.first_token_at
        if duration <= 0.0:
            return None
        return (output_tokens - 1) / duration


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_seconds: float = 0.0
    timed_input_tokens: int = 0
    timed_output_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    def add(self, result: ResponseResult, elapsed: float) -> None:
        self.requests += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.total_tokens += result.total_tokens
        self.request_seconds += elapsed

        if result.prefill_seconds is not None:
            self.timed_input_tokens += result.input_tokens
            self.prefill_seconds += result.prefill_seconds
        if result.decode_seconds is not None:
            self.timed_output_tokens += result.output_tokens
            self.decode_seconds += result.decode_seconds

    def output_tokens_per_second(self) -> Optional[float]:
        if self.request_seconds <= 0.0:
            return None
        return self.output_tokens / self.request_seconds

    def prefill_tokens_per_second(self) -> Optional[float]:
        if self.prefill_seconds <= 0.0:
            return None
        return self.timed_input_tokens / self.prefill_seconds

    def decode_tokens_per_second(self) -> Optional[float]:
        if self.decode_seconds <= 0.0:
            return None
        return self.timed_output_tokens / self.decode_seconds


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

                lines: queue.Queue[object] = queue.Queue()
                stream_finished = object()

                def read_stream() -> None:
                    try:
                        for line in response.iter_lines():
                            lines.put(line)
                    except BaseException as error:
                        lines.put(error)
                    finally:
                        lines.put(stream_finished)

                reader = threading.Thread(
                    target=read_stream,
                    name="adi-response-reader",
                    daemon=True,
                )
                reader.start()
                try:
                    while True:
                        try:
                            item = lines.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if item is stream_finished:
                            break
                        if isinstance(item, BaseException):
                            raise item
                        line = str(item)
                        if line == "":
                            dispatch()
                        elif line.startswith(":"):
                            continue
                        elif line.startswith("event:"):
                            event_name = line[6:].lstrip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    dispatch()
                except BaseException:
                    try:
                        response.close()
                    except Exception:
                        pass
                    reader.join(timeout=5.0)
                    raise
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


class WindowsKillOnCloseJob:
    """Own a Windows job that kills its processes when this client disappears."""

    KILL_ON_JOB_CLOSE = 0x00002000
    EXTENDED_LIMIT_INFORMATION = 9
    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100

    def __init__(self, handle: int, kernel32: Any) -> None:
        self.handle = handle
        self.kernel32 = kernel32

    @classmethod
    def create(cls) -> "WindowsKillOnCloseJob":
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = cls.KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            cls.EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(handle)
            raise error
        return cls(handle, kernel32)

    def assign(self, process_id: int) -> None:
        import ctypes
        from ctypes import wintypes

        self.kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self.kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

        process = self.kernel32.OpenProcess(
            self.PROCESS_TERMINATE | self.PROCESS_SET_QUOTA,
            False,
            process_id,
        )
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel32.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel32.CloseHandle(process)

    def close(self) -> None:
        if self.handle:
            self.kernel32.CloseHandle(self.handle)
            self.handle = 0


class ManagedAdiServer:
    def __init__(
        self,
        process: subprocess.Popen[str],
        log_tail: deque[str],
        host: str,
        port: int,
        kill_job: Optional[WindowsKillOnCloseJob] = None,
    ) -> None:
        self.process = process
        self.log_tail = log_tail
        self.host = host
        self.port = port
        self.kill_job = kill_job
        self.listening = False
        self.log_thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{host_for_connect(self.host)}:{self.port}"

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
        if port != 0 and port_is_open(host_for_connect(host), port):
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
        kill_job: Optional[WindowsKillOnCloseJob] = None
        process: Optional[subprocess.Popen[str]] = None
        try:
            if os.name == "nt":
                kill_job = WindowsKillOnCloseJob.create()
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
            if kill_job is not None:
                kill_job.assign(process.pid)
        except OSError as error:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)
            if kill_job is not None:
                kill_job.close()
            raise ClientError(f"cannot start ADI: {error}") from error

        log_tail: deque[str] = deque(maxlen=40)
        server = cls(process, log_tail, host, port, kill_job)

        def drain_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                stripped = line.rstrip("\r\n")
                log_tail.append(stripped)
                detected_port = parse_listening_port(stripped)
                if detected_port is not None:
                    server.port = detected_port
                    server.listening = True
                if show_log:
                    console.print(f"[adi] {stripped}", style="dim", markup=False)

        server.log_thread = threading.Thread(
            target=drain_output,
            name="adi-log-reader",
            daemon=True,
        )
        server.log_thread.start()
        try:
            server.wait_until_ready(startup_timeout)
        except BaseException:
            server.stop()
            raise
        return server

    def wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                tail = "\n".join(self.log_tail)
                detail = f"\n{tail}" if tail else ""
                raise ClientError(f"ADI exited with status {return_code}{detail}")
            if self.listening:
                return
            time.sleep(0.1)
        tail = "\n".join(self.log_tail)
        detail = f"\nRecent ADI output:\n{tail}" if tail else ""
        raise ClientError(f"ADI did not listen within {timeout:g} seconds{detail}")

    def stop(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
        finally:
            if self.kill_job is not None:
                self.kill_job.close()
                self.kill_job = None
        if self.log_thread is not None:
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
    timings = body.get("timings")
    prefill_seconds = decode_seconds = None
    prefill_rate = decode_rate = None
    if isinstance(timings, dict):
        prompt_ms = nonnegative_number_or_none(timings.get("prompt_ms"))
        predicted_ms = nonnegative_number_or_none(timings.get("predicted_ms"))
        if prompt_ms is not None:
            prefill_seconds = prompt_ms / 1000.0
        if predicted_ms is not None:
            decode_seconds = predicted_ms / 1000.0
        prefill_rate = nonnegative_number_or_none(
            timings.get("prompt_per_second")
        )
        decode_rate = nonnegative_number_or_none(
            timings.get("predicted_per_second")
        )
        if (
            prefill_rate is None
            and prefill_seconds is not None
            and prefill_seconds > 0
        ):
            prefill_rate = input_tokens / prefill_seconds
        if (
            decode_rate is None
            and decode_seconds is not None
            and decode_seconds > 0
        ):
            decode_rate = output_tokens / decode_seconds
    return ResponseResult(
        text="".join(text_parts),
        status=str(body.get("status", "completed")),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens or input_tokens + output_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        prefill_tokens_per_second=prefill_rate,
        decode_tokens_per_second=decode_rate,
        raw=body,
    )


class ReasoningRenderer:
    def __init__(
        self,
        emit: Optional[Callable[[str, bool], None]] = None,
    ) -> None:
        # ADI prompts assistant responses from "<think>\\n", so the first streamed
        # characters are reasoning content unless a prior close token is received.
        self.in_reasoning = True
        self.pending = ""
        self.emit = emit

    def feed(self, text: str, *, final: bool = False) -> None:
        buffered = self.pending + text
        self.pending = ""
        while buffered:
            if self.in_reasoning:
                close_index = buffered.lower().find(THINK_CLOSE_TAG)
                if close_index == -1:
                    carry = THINK_SUFFIX_GUARD
                    if not final and len(buffered) > carry:
                        emit = buffered[:-carry]
                        self._print_dim(emit)
                        buffered = buffered[-carry:]
                        continue
                    if final:
                        self._print_dim(buffered)
                    else:
                        self.pending = buffered
                    break
                self._print_dim(buffered[:close_index])
                buffered = buffered[close_index + len(THINK_CLOSE_TAG):]
                self.in_reasoning = False
                continue

            open_index = buffered.lower().find(THINK_OPEN_TAG)
            if open_index == -1:
                carry = THINK_SUFFIX_GUARD
                if not final and len(buffered) > carry:
                    emit = buffered[:-carry]
                    self._print(emit)
                    buffered = buffered[-carry:]
                    continue
                if final:
                    self._print(buffered)
                else:
                    self.pending = buffered
                break

            self._print(buffered[:open_index])
            self.in_reasoning = True
            buffered = buffered[open_index + len(THINK_OPEN_TAG):]

    def _print(self, text: str) -> None:
        if text:
            if self.emit is not None:
                self.emit(text, False)
                return
            console.print(
                text,
                end="",
                markup=False,
                highlight=False,
                soft_wrap=True,
            )

    def _print_dim(self, text: str) -> None:
        if text:
            if self.emit is not None:
                self.emit(text, True)
                return
            console.print(
                text,
                end="",
                style=DIM_REASONING_STYLE,
                markup=False,
                highlight=False,
                soft_wrap=True,
            )


def integer_or_zero(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def nonnegative_number_or_none(value: object) -> Optional[float]:
    if type(value) not in {int, float}:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0.0 else None


def host_for_connect(host: str) -> str:
    return "127.0.0.1" if host == "0.0.0.0" else host


def parse_listening_port(line: str) -> Optional[int]:
    prefix = "adi: listening on http://"
    suffix = "/v1/responses"
    if not line.startswith(prefix) or not line.endswith(suffix):
        return None
    authority = line[len(prefix):-len(suffix)]
    _, separator, port_text = authority.rpartition(":")
    if not separator or not port_text.isdigit():
        return None
    port = int(port_text)
    return port if 1 <= port <= 65535 else None


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


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


class StreamingStatus:
    """A live footer rendered beneath streamed response text."""

    def __init__(self, metrics: TurnMetrics) -> None:
        self.metrics = metrics

    def __rich__(self) -> Text:
        if self.metrics.first_token_at is None:
            message = f"PREFILLING  •  {self.metrics.elapsed():.1f}s elapsed"
        else:
            tokens = self.metrics.streamed_tokens
            parts = ["GENERATING", plural(tokens, "token")]
            current_rate = self.metrics.current_tokens_per_second()
            if current_rate is None:
                parts.append("measuring rate…")
            else:
                parts.append(f"{current_rate:.2f} tok/s current")
            average_rate = self.metrics.average_tokens_per_second()
            if average_rate is not None and tokens > len(
                self.metrics.recent_token_times
            ):
                parts.append(f"{average_rate:.2f} tok/s avg")
            parts.append(f"{self.metrics.elapsed():.1f}s elapsed")
            message = "  •  ".join(parts)
        return Text(
            f"  {message}  ",
            style=STATUS_BAR_STYLE,
            justify="left",
            overflow="ellipsis",
            no_wrap=True,
        )


class StreamingDisplay:
    """Own the incomplete response line and status as one Rich live region."""

    def __init__(self, metrics: TurnMetrics) -> None:
        self.metrics = metrics
        self.partial_line = Text()
        self.partial_reasoning: Optional[bool] = None
        self.reasoning_started = False
        self.answer_started = False

    def __rich__(self) -> Group:
        return Group(
            self.partial_line,
            Text(""),
            StreamingStatus(self.metrics).__rich__(),
        )

    def _start_phase(self, reasoning: bool) -> None:
        if reasoning:
            if not self.reasoning_started:
                console.print(Text("  REASONING  ", style=REASONING_LABEL_STYLE))
                self.reasoning_started = True
            return
        if self.answer_started:
            return
        if self.reasoning_started:
            console.print()
        console.print(Text("  ANSWER  ", style=ANSWER_LABEL_STYLE))
        self.answer_started = True

    def _start_line(self, reasoning: bool) -> None:
        self._start_phase(reasoning)
        self.partial_reasoning = reasoning
        if reasoning:
            self.partial_line.append("  │ ", style=REASONING_GUTTER_STYLE)

    def _commit_line(self) -> None:
        console.print(self.partial_line, highlight=False, soft_wrap=True)
        self.partial_line = Text()
        self.partial_reasoning = None

    def write(self, text: str, reasoning: bool) -> None:
        style = DIM_REASONING_STYLE if reasoning else None
        while text:
            newline = text.find("\n")
            segment = text if newline == -1 else text[:newline]
            if segment:
                if self.partial_reasoning is None:
                    self._start_line(reasoning)
                elif self.partial_reasoning != reasoning:
                    self._commit_line()
                    self._start_line(reasoning)
                self.partial_line.append(segment, style=style)
            if newline == -1:
                return
            if self.partial_reasoning is None and reasoning:
                self._start_line(True)
            self._commit_line()
            text = text[newline + 1 :]

    def finish(self) -> None:
        if self.partial_line:
            console.print(
                self.partial_line,
                end="",
                highlight=False,
                soft_wrap=True,
            )
            self.partial_line = Text()
            self.partial_reasoning = None


def plural(count: int, noun: str) -> str:
    return f"{count:,} {noun if count == 1 else noun + 's'}"


def print_usage(
    result: ResponseResult,
    totals: UsageTotals,
    metrics: TurnMetrics,
) -> None:
    elapsed = metrics.elapsed()
    totals.add(result, elapsed)
    has_server_timings = any(
        value is not None
        for value in (
            result.prefill_seconds,
            result.decode_seconds,
            result.prefill_tokens_per_second,
            result.decode_tokens_per_second,
        )
    )
    if has_server_timings:
        prefill_parts = [f"{result.input_tokens:,} input tokens"]
        if result.prefill_seconds is not None:
            prefill_parts.append(f"{result.prefill_seconds:.2f}s")
        if result.prefill_tokens_per_second is not None:
            prefill_parts.append(
                f"{result.prefill_tokens_per_second:.2f} tok/s"
            )
        decode_parts = [f"{result.output_tokens:,} output tokens"]
        if result.decode_seconds is not None:
            decode_parts.append(f"{result.decode_seconds:.2f}s")
        if result.decode_tokens_per_second is not None:
            decode_parts.append(f"{result.decode_tokens_per_second:.2f} tok/s")
        console.print(
            "  prefill  " + "  •  ".join(prefill_parts),
            style="dim",
            markup=False,
        )
        console.print(
            "  decode   " + "  •  ".join(decode_parts),
            style="dim",
            markup=False,
        )
        request_parts = [f"{elapsed:.2f}s total"]
        first_token = metrics.time_to_first_token()
        if first_token is not None:
            request_parts.append(f"{first_token:.2f}s TTFT")
        request_parts.append(f"{totals.total_tokens:,} session tokens")
        console.print(
            "  request  " + "  •  ".join(request_parts),
            style="dim",
            markup=False,
        )
        return

    # Older servers do not return authoritative phase timings. Retain useful
    # client-observed metrics when connecting to one of them.
    parts = [
        f"{result.output_tokens:,} output",
        f"{result.input_tokens:,} input",
    ]
    decode_rate = metrics.decode_tokens_per_second(result.output_tokens)
    if decode_rate is not None:
        parts.append(f"{decode_rate:.2f} tok/s decode")
    elif elapsed > 0.0 and result.output_tokens:
        parts.append(f"{result.output_tokens / elapsed:.2f} tok/s end-to-end")
    first_token = metrics.time_to_first_token()
    if first_token is not None:
        parts.append(f"{first_token:.1f}s TTFT")
    parts.extend(
        [
            f"{elapsed:.1f}s total",
            f"{totals.total_tokens:,} session tokens",
        ]
    )
    console.print("  " + "  •  ".join(parts), style="dim", markup=False)


def print_session_stats(totals: UsageTotals) -> None:
    if totals.requests == 0:
        console.print("No completed requests in this session.", style="dim")
        return
    summary = [
        plural(totals.requests, "request"),
        f"{totals.input_tokens:,} input",
        f"{totals.output_tokens:,} output",
        f"{totals.total_tokens:,} total tokens",
        f"{totals.request_seconds:.1f}s request time",
    ]
    console.print("  •  ".join(summary), style="dim", markup=False)
    prefill_rate = totals.prefill_tokens_per_second()
    if prefill_rate is not None:
        console.print(
            f"  prefill  {totals.timed_input_tokens:,} input tokens  •  "
            f"{totals.prefill_seconds:.2f}s  •  {prefill_rate:.2f} tok/s",
            style="dim",
            markup=False,
        )
    decode_rate = totals.decode_tokens_per_second()
    if decode_rate is not None:
        console.print(
            f"  decode   {totals.timed_output_tokens:,} output tokens  •  "
            f"{totals.decode_seconds:.2f}s  •  {decode_rate:.2f} tok/s",
            style="dim",
            markup=False,
        )
    if prefill_rate is None and decode_rate is None:
        rate = totals.output_tokens_per_second()
        if rate is not None:
            console.print(
                f"  {rate:.2f} output tok/s end-to-end",
                style="dim",
                markup=False,
            )


def incomplete_reason(result: ResponseResult) -> Optional[str]:
    if result.status == "completed":
        return None
    details = result.raw.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    if reason == "max_output_tokens":
        return "Output limit reached; the response may be unfinished."
    if isinstance(reason, str) and reason:
        return f"Response ended early: {reason.replace('_', ' ')}."
    return f"Response ended with status {result.status}."


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


def read_user_input_line() -> str:
    if readline is None or not sys.stdin.isatty() or not sys.stdout.isatty():
        return console.input("[bold green]user[/bold green] > ").strip()
    return input(READLINE_PROMPT).strip()


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
        "  /stats                Show cumulative usage and timing\n"
        "  /settings             Show endpoint and generation settings"
    )


def settings_summary(settings: GenerationSettings, stream: bool) -> str:
    limit = (
        f"{settings.max_output_tokens:,} max output"
        if settings.max_output_tokens is not None
        else "context-limited output"
    )
    sampling = (
        "greedy"
        if settings.temperature == 0.0
        else f"temperature {settings.temperature:g}  •  top-p {settings.top_p:g}"
    )
    return (
        f"{'streaming' if stream else 'non-streaming'}  •  {sampling}  •  "
        f"seed {settings.seed:,}  •  {limit}"
    )


def print_settings(
    client: AdiResponsesClient,
    settings: GenerationSettings,
    stream: bool,
) -> None:
    console.print(client.responses_url, style="cyan", markup=False)
    console.print(settings_summary(settings, stream), style="dim", markup=False)
    if settings.model_id:
        console.print(f"model {settings.model_id}", style="dim", markup=False)


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
    metrics = TurnMetrics()
    show_live_status = stream and bool(getattr(console, "is_terminal", False))
    streaming_display = StreamingDisplay(metrics) if show_live_status else None
    renderer = ReasoningRenderer(
        streaming_display.write if streaming_display is not None else None
    )
    printed_delta = False

    def on_delta(delta: str) -> None:
        nonlocal printed_delta
        if delta:
            metrics.observe_token()
        printed_delta = printed_delta or bool(delta)
        renderer.feed(delta)

    live_context = (
        Live(
            streaming_display,
            console=console,
            refresh_per_second=4,
            transient=True,
        )
        if show_live_status
        else nullcontext()
    )
    try:
        with live_context:
            result = client.create(
                conversation.request_messages(user_text),
                settings,
                stream=stream,
                on_delta=on_delta,
            )
            metrics.finish()
            if streaming_display is not None:
                if not printed_delta and result.text:
                    renderer.feed(result.text, final=True)
                else:
                    renderer.feed("", final=True)
    except KeyboardInterrupt:
        if streaming_display is not None:
            renderer.feed("", final=True)
            streaming_display.finish()
        if printed_delta:
            console.print()
        console.print("[yellow]Generation interrupted; the turn was not saved.[/yellow]")
        raise
    except ClientError:
        if streaming_display is not None:
            renderer.feed("", final=True)
            streaming_display.finish()
        if printed_delta:
            console.print()
        raise
    if stream:
        if streaming_display is not None:
            streaming_display.finish()
        else:
            if not printed_delta and result.text:
                renderer.feed(result.text, final=True)
            else:
                renderer.feed("", final=True)
        console.print()
    else:
        renderer.feed(result.text, final=True)
        console.print()
    if not result.text:
        console.print("[yellow]ADI returned no output text.[/yellow]")
    conversation.commit(user_text, result.text)
    reason = incomplete_reason(result)
    if reason is not None:
        console.print(reason, style="yellow", markup=False)
    print_usage(result, totals, metrics)
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
    console.print("[bold green]ADI chat[/bold green]")
    console.print(
        f"{client.responses_url}  •  {settings_summary(settings, stream)}",
        style="dim",
        markup=False,
    )
    if conversation.messages:
        console.print(
            f"Loaded {plural(len(conversation.messages) // 2, 'prior turn')}.",
            style="dim",
            markup=False,
        )
    console.print(
        "Type /help for commands; Ctrl+C, Ctrl+D, or /exit quits.",
        style="dim",
    )
    while True:
        try:
            line = read_user_input_line()
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            console.print()
            return
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
                print_session_stats(totals)
                continue
            if command == "/settings":
                print_settings(client, settings, stream)
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
        0,
        min=0,
        max=65535,
        help="Port passed to ADI. The default 0 lets OS choose a free local port.",
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
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        min=1,
        help=(
            "Optional max output tokens per turn. Omit to use the full remaining "
            "context budget."
        ),
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
            endpoint = f"{host}:automatic" if port == 0 else f"{host}:{port}"
            console.print(
                f"[dim]Starting {executable} on {endpoint}; "
                "model loading may take a while.[/dim]"
            )
            managed_server = ManagedAdiServer.start(
                executable,
                model,
                host,
                port,
                threads=threads,
                isa=isa,
                startup_timeout=startup_timeout,
                show_log=server_log,
            )
            base_url = managed_server.base_url
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
    configure_windows_utf8_stdio()
    configure_windows_console_signals()
    console = Console()
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()
