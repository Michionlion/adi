# Responses API

ADI exposes one inference route:

```text
POST /v1/responses
```

The initial server accepts JSON bodies with:

- `input`: a string or an array of text-only messages, prior `function_call`
  items, and `function_call_output` items;
- `instructions`: optional text folded into the leading system message;
- `model`: optional; when present it must equal the loaded model's
  `general.name`;
- `max_output_tokens`: optional positive integer; omitted defaults to the remaining
  context budget after the input prompt (`context_length - input_tokens`).
- `temperature`: `0` for greedy decoding or a number in `[0.0001, 2]`;
- `top_p`: number in `(0, 1]`;
- `seed`: non-negative integer;
- `stream`: boolean; `true` returns Responses API server-sent events;
- `tools`: up to 128 flat Responses API function definitions;
- `tool_choice`: `auto` or `none`;
- `parallel_tool_calls`: boolean, defaulting to `true` when tools are present.

Only non-strict function tools are supported. Built-in, custom, MCP, hosted,
and strict function tools are rejected explicitly. ADI reports calls but never
executes them: the client must run the function and submit its result as a
`function_call_output`. `previous_response_id` is not stored; send the full
input history on each request. Image content and other API routes are also
rejected. Requests are handled one at a time so one process never
oversubscribes the CPU while a generation is active.

Text messages are rendered with the validated checkpoint chat template's
default thinking-enabled mode. Message content is trimmed, a system message
must be first, and prior assistant `<think>...</think>` content is omitted from
history according to the Qwen3.5 conversation rules.

Function definitions are translated from the Responses API's flat shape into
the checkpoint template's native Qwen tool shape. A successful model call is
returned as a standard output item:

```json
{
  "type": "function_call",
  "id": "fc_...",
  "call_id": "call_...",
  "name": "calculator",
  "arguments": "{\"operation\":\"add\",\"a\":2,\"b\":3}",
  "status": "completed"
}
```

Continue the turn by resending the preceding input and response output, then
append:

```json
{
  "type": "function_call_output",
  "call_id": "call_...",
  "output": "{\"result\":5}"
}
```

Example:

```bash
curl http://127.0.0.1:9932/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Mach-1-Additive-35B",
    "input": "Write one sentence about additive models.",
    "max_output_tokens": 32,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
  }'
```

Successful responses use the Responses API `response` object with assistant
`message` and/or `function_call` output items plus input/output/total token
usage. ADI also returns a top-level `timings` object compatible with llama.cpp's
performance fields:

```json
{
  "timings": {
    "prompt_n": 128,
    "prompt_ms": 3200.0,
    "prompt_per_token_ms": 25.0,
    "prompt_per_second": 40.0,
    "predicted_n": 32,
    "predicted_ms": 8000.0,
    "predicted_per_token_ms": 250.0,
    "predicted_per_second": 4.0
  }
}
```

Prompt timing measures model prefill and excludes request queueing and
tokenization. Prediction timing starts after prefill and ends when generation
finishes. Streaming responses carry these timings in the terminal response
object and emit the normal lifecycle from
`response.created` through `response.completed`, including UTF-8-safe
`response.output_text.delta` events. Function calls additionally emit
`response.function_call_arguments.delta` and
`response.function_call_arguments.done`; ADI buffers one complete call before
emitting its canonical argument JSON. Hitting `max_output_tokens` instead
returns status `incomplete`, `incomplete_details.reason` set to
`max_output_tokens`,
and a terminal `response.incomplete` event. Errors use an `error` object and
an HTTP 4xx status, or an SSE `error` event after streaming headers have been
sent.

Header reads have a 15-second deadline, body reads have a 30-second deadline,
socket writes have a 15-second no-progress deadline, and a request has a
30-minute wall-clock deadline. A disconnected peer cancels generation at the
next token boundary. These bounds keep the intentionally serialized inference
slot from being held indefinitely by network I/O.
