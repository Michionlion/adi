# Python chat client

`tools/adi_chat.py` is a small cross-platform client for ADI's Responses API. It
can start `adi serve`, wait for the model to finish loading, stream a
conversation in the terminal, persist message history, and stop the child
server when the client exits.

The client intentionally uses a tiny in-process conversation harness rather
than a general coding-agent framework. ADI currently accepts text messages but
rejects tool definitions, so a full tool-calling agent loop would add another
runtime without adding usable capabilities.

## Quick start with uv

The script contains PEP 723 dependency metadata, so a recent `uv` can install
its three Python dependencies automatically:

```bash
uv run tools/adi_chat.py \
  --model models/Mach-1-Additive-35B.gguf
```

On Windows PowerShell:

```powershell
uv run tools/adi_chat.py `
  --adi build/Release/adi.exe `
  --model models/Mach-1-Additive-35B.gguf
```

The executable is auto-detected from `PATH`, `ADI_EXE`, and the usual CMake
build locations. Model loading may take several minutes; the default startup
timeout is ten minutes.

Without `uv`, install the dependencies and invoke the script normally:

```powershell
py -m pip install httpx typer rich
py tools/adi_chat.py --adi build/Release/adi.exe --model MODEL.gguf
```

## Connect to an existing server

```bash
uv run tools/adi_chat.py --connect http://127.0.0.1:9932
```

`--connect` also accepts a base ending in `/v1` or the complete
`/v1/responses` endpoint.

## Useful options

```text
--port 0                 choose an unused local port (default)
--threads 8              set ADI_THREADS for the child process
--isa scalar             set ADI_CPU_ISA
--system "..."           install a system prompt
--session chat.json      load and save conversation history
--max-tokens 512         set the output limit for each turn
--no-stream              use a normal JSON response instead of SSE
--prompt "Hello"         send one request and exit
--server-log             mirror ADI's output into the terminal
```

The API `model` field is omitted by default because ADI treats it as optional.
Use `--model-id NAME` when a caller needs to send it explicitly.

## Interactive commands

```text
/help                 show commands
/reset                clear user/assistant history
/system [TEXT]        show or replace the system prompt
/history              show abbreviated history
/save PATH            save a session
/load PATH            load a session
/paste                enter a multi-line message; finish with a single `.`
/stats                show cumulative token usage
/exit                 exit and stop the ADI child process
```

A turn is committed to history only after a complete response. Interrupting a
stream with Ctrl+C closes the HTTP request, leaves the local history unchanged,
and allows ADI to observe the disconnect through its normal cancellation path.
