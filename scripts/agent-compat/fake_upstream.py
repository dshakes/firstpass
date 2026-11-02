#!/usr/bin/env python3
"""Recording fake upstream for agent-compat testing.

Listens on an ephemeral port, records every request (method, path, headers
minus secrets, body shape), and returns protocol-correct responses for:
  - POST /v1/messages           Anthropic Messages (non-streaming + streaming SSE)
  - POST /v1/messages/count_tokens   Anthropic count-tokens
  - POST /v1/chat/completions   OpenAI Chat Completions (non-streaming + streaming)
  - POST /v1/responses          OpenAI Responses API (non-streaming + streaming)
  - GET  /v1/models             OpenAI model list

Usage (invoked by run_compat.sh):
  python3 fake_upstream.py <PORT>
  python3 fake_upstream.py  # prints port on stdout, then listens
"""

import http.server
import json
import sys
import threading

# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

ANTHROPIC_NON_STREAM = json.dumps(
    {
        "id": "msg_compat_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "I am the fake upstream."}],
        "model": "claude-haiku-20240307",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 6},
    }
)

# Minimal SSE: message_start -> content_block_start -> content_block_delta
# -> content_block_stop -> message_delta -> message_stop
ANTHROPIC_SSE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_compat_1","type":"message",'
    '"role":"assistant","content":[],"model":"claude-haiku-20240307",'
    '"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"Fake response."}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
    '"stop_sequence":null},"usage":{"output_tokens":2}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)

# Streaming SSE with a tool_use turn (rung 0) followed by a text turn (rung 1).
# Claude Code typically sends multi-turn conversations; this exercises the proxy's
# SSE-tapping path without needing a real model.
ANTHROPIC_TOOL_SSE = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_compat_2","type":"message",'
    '"role":"assistant","content":[],"model":"claude-haiku-20240307",'
    '"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":15,"output_tokens":0}}}\n\n'
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"tool_use","id":"toolu_01","name":"bash","input":{}}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"input_json_delta","partial_json":"{}\\""}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use",'
    '"stop_sequence":null},"usage":{"output_tokens":5}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)

ANTHROPIC_COUNT_TOKENS = json.dumps({"input_tokens": 42})

OPENAI_NON_STREAM = json.dumps(
    {
        "id": "chatcmpl-compat1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I am the fake upstream."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }
)

OPENAI_SSE = (
    'data: {"id":"chatcmpl-compat1","object":"chat.completion.chunk",'
    '"created":1700000000,"model":"gpt-3.5-turbo",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Fake"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-compat1","object":"chat.completion.chunk",'
    '"created":1700000000,"model":"gpt-3.5-turbo",'
    '"choices":[{"index":0,"delta":{"content":" response."},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-compat1","object":"chat.completion.chunk",'
    '"created":1700000000,"model":"gpt-3.5-turbo",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n'
    "data: [DONE]\n\n"
)

OPENAI_MODELS = json.dumps(
    {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            },
            {"id": "gpt-4", "object": "model", "created": 0, "owned_by": "openai"},
        ],
    }
)

# OpenAI Responses API streaming events
RESPONSES_SSE = (
    "event: response.created\n"
    'data: {"type":"response.created","response":{"id":"resp_compat1",'
    '"object":"realtime.response","status":"in_progress","output":[]}}\n\n'
    "event: response.output_item.added\n"
    'data: {"type":"response.output_item.added","response_id":"resp_compat1",'
    '"output_index":0,"item":{"id":"item_1","object":"realtime.item",'
    '"type":"message","status":"in_progress","role":"assistant","content":[]}}\n\n'
    "event: response.content_part.added\n"
    'data: {"type":"response.content_part.added","item_id":"item_1",'
    '"output_index":0,"content_index":0,"part":{"type":"text","text":""}}\n\n'
    "event: response.text.delta\n"
    'data: {"type":"response.text.delta","item_id":"item_1",'
    '"output_index":0,"content_index":0,"delta":"Fake response."}\n\n'
    "event: response.text.done\n"
    'data: {"type":"response.text.done","item_id":"item_1",'
    '"output_index":0,"content_index":0,"text":"Fake response."}\n\n'
    "event: response.output_item.done\n"
    'data: {"type":"response.output_item.done","response_id":"resp_compat1",'
    '"output_index":0,"item":{"id":"item_1","status":"completed"}}\n\n'
    "event: response.done\n"
    'data: {"type":"response.done","response":{"id":"resp_compat1","status":"completed",'
    '"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}}\n\n'
)

RESPONSES_NON_STREAM = json.dumps(
    {
        "id": "resp_compat1",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Fake response."}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
)

# ---------------------------------------------------------------------------
# Recorded requests (thread-safe list)
# ---------------------------------------------------------------------------

recorded = []
_lock = threading.Lock()
# Mutable container for the log path so the __main__ block can set it
# without needing a `global` declaration (which is only valid in functions).
_log_path: list = [None]


def record(entry):
    with _lock:
        recorded.append(entry)
        # Flush after every request so the shell harness can read mid-run.
        if _log_path[0]:
            try:
                with open(_log_path[0], "w") as f:
                    f.write(json.dumps(recorded, indent=2))
            except OSError:
                pass


def dump_log(path=None):
    with _lock:
        data = list(recorded)
    text = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(text)
    else:
        print(text)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

SKIP_HEADERS = frozenset(
    [
        "x-api-key",
        "authorization",
        "x-firstpass-key",
        "cookie",
        "set-cookie",
    ]
)


def redact_headers(headers):
    """Return headers dict with auth values redacted."""
    out = {}
    for k, v in headers.items():
        key = k.lower()
        if key in SKIP_HEADERS:
            out[key] = "<redacted>"
        else:
            out[key] = v
    return out


def body_shape(raw):
    """Top-level keys + stream flag from a JSON body; 'binary' if not JSON."""
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {"type": "non-object"}
        shape = {
            "keys": sorted(parsed.keys()),
            "stream": bool(parsed.get("stream", False)),
        }
        if "model" in parsed:
            shape["model"] = parsed["model"]
        return shape
    except Exception:
        return {"type": "binary", "len": len(raw)}


class FakeUpstreamHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default stderr logging
        pass

    def send_json(self, code, body):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # Prevent reqwest (and other HTTP/1.1 clients) from pooling the connection
        # across sequential requests; Python's BaseHTTPRequestHandler defaults to
        # HTTP/1.0 semantics, but explicit is better than implicit here.
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def send_sse(self, code, body):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        raw = b""
        entry = {
            "method": "GET",
            "path": self.path,
            "headers": redact_headers(dict(self.headers)),
            "body_shape": {},
        }
        record(entry)

        if self.path.split("?")[0] == "/v1/models":
            self.send_json(200, OPENAI_MODELS)
        elif self.path == "/healthz":
            self.send_json(200, '{"status":"ok","service":"fake-upstream"}')
        else:
            self.send_json(404, '{"error":"not found"}')

    def do_POST(self):
        raw = self._read_body()
        is_stream = b'"stream":true' in raw or b'"stream": true' in raw
        entry = {
            "method": "POST",
            "path": self.path,
            "headers": redact_headers(dict(self.headers)),
            "body_shape": body_shape(raw),
        }
        record(entry)

        path = self.path.split("?")[0]

        if path == "/v1/messages":
            if is_stream:
                self.send_sse(200, ANTHROPIC_SSE)
            else:
                self.send_json(200, ANTHROPIC_NON_STREAM)

        elif path == "/v1/messages/count_tokens":
            self.send_json(200, ANTHROPIC_COUNT_TOKENS)

        elif path == "/v1/chat/completions":
            if is_stream:
                self.send_sse(200, OPENAI_SSE)
            else:
                self.send_json(200, OPENAI_NON_STREAM)

        elif path == "/v1/responses":
            if is_stream:
                self.send_sse(200, RESPONSES_SSE)
            else:
                self.send_json(200, RESPONSES_NON_STREAM)

        else:
            self.send_json(404, json.dumps({"error": f"unknown path {path}"}))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    log_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Mutate the container so record() picks up the path on every flush call.
    _log_path[0] = log_path

    server = http.server.HTTPServer(("127.0.0.1", port), FakeUpstreamHandler)
    actual_port = server.server_address[1]

    # Print the port first so the caller can read it before blocking.
    print(actual_port, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if log_path:
            dump_log(log_path)
        else:
            dump_log()
