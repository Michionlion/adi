# Responses API

ADI exposes one inference route:

```text
POST /v1/responses
```

The initial server accepts JSON bodies with:

- `input`: a string or an array of text-only `system`, `user`, and `assistant`
  messages;
- `model`: optional response label;
- `max_output_tokens`: non-negative integer;
- `temperature`: non-negative number (`0` selects greedy decoding);
- `top_p`: number in `(0, 1]`;
- `seed`: non-negative integer;
- `stream`: boolean; `true` returns Responses API server-sent events.

Image content, tools, and other API routes are rejected explicitly. Requests
are handled one at a time so one process never oversubscribes the CPU while a
generation is active.

Example:

```bash
curl http://127.0.0.1:8080/v1/responses \
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

Successful responses use the Responses API `response` object with one
assistant `message`, one `output_text` content item, and input/output/total
token usage. Streaming responses emit the normal lifecycle from
`response.created` through `response.completed`, including UTF-8-safe
`response.output_text.delta` events. Errors use an `error` object and an HTTP
4xx status, or an SSE `error` event after streaming headers have been sent.
