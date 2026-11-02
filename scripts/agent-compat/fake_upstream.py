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
import os
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
    '"delta":{"type":"input_json_delta","partial_json":"{}"}}\n\n'
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
                # Write-then-rename so a concurrent reader never sees a truncated file.
                tmp = _log_path[0] + ".tmp"
                with open(tmp, "w") as f:
                    f.write(json.dumps(recorded, indent=2))
                os.replace(tmp, _log_path[0])
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
# Enforce-mode script (FAKE_SCRIPT=enforce). Deterministic, per-model behaviour so a real CLI
# drives a real escalation through the proxy's gates:
#   * cheap model (haiku / *-mini): its FIRST reply is bad -- a tool_use with args that violate
#     the declared schema when the request declares tools (Claude Code), else an empty reply
#     (aider). Every later cheap reply is valid.
#   * top model: always valid, unless FAKE_TOP_FAILS=1 (negative control for the harness).
#   * a request whose last message is a tool_result gets plain text from any model.
# The chosen behaviour is stamped on the request log entry as "scripted".
# ---------------------------------------------------------------------------

SCRIPT_ON = os.environ.get("FAKE_SCRIPT") == "enforce"
TOP_FAILS = os.environ.get("FAKE_TOP_FAILS") == "1"
_cheap_seen = {"anthropic": 0, "openai": 0}


def is_cheap(model):
    return "haiku" in model or "mini" in model


def declared_tools(parsed):
    """[(name, schema)] from an Anthropic (input_schema) or OpenAI (function.parameters) body."""
    out = []
    for t in parsed.get("tools") or []:
        if "function" in t:
            out.append(
                (t["function"].get("name"), t["function"].get("parameters") or {})
            )
        else:
            out.append((t.get("name"), t.get("input_schema") or {}))
    return out


def valid_args(schema):
    return {k: "echo fp-enforce-ok" for k in schema.get("required", [])}


def pick_tool(tools):
    for name, schema in tools:
        if name == "Bash" and schema.get("required"):
            return name, schema
    for name, schema in tools:
        if schema.get("required"):
            return name, schema
    return None, {}


def last_is_tool_result(parsed):
    msgs = parsed.get("messages") or []
    if not msgs:
        return False
    last = msgs[-1]
    if last.get("role") == "tool":
        return True
    c = last.get("content")
    return isinstance(c, list) and any(b.get("type") == "tool_result" for b in c)


def decide(api, parsed):
    """Return (scripted-label, kind, tool_name, tool_args). kind: text|empty|tool."""
    model = parsed.get("model", "")
    if last_is_tool_result(parsed):
        return "text-after-tool-result", "text", None, None
    name, schema = pick_tool(declared_tools(parsed))
    bad = False
    if is_cheap(model):
        _cheap_seen[api] += 1
        bad = _cheap_seen[api] == 1
    elif TOP_FAILS:
        bad = True
    tag = "cheap" if is_cheap(model) else "top"
    if name:
        if bad:
            return f"{tag}-bad-tool-args", "tool", name, {}
        return f"{tag}-valid-tool", "tool", name, valid_args(schema)
    if bad:
        return f"{tag}-empty", "empty", None, None
    return f"{tag}-text", "text", None, None


def scripted_anthropic(parsed, kind, name, args, stream):
    model = parsed.get("model", "claude")
    if kind == "tool":
        blocks = [{"type": "tool_use", "id": "toolu_fp1", "name": name, "input": args}]
        stop = "tool_use"
    elif kind == "empty":
        blocks, stop = [{"type": "text", "text": ""}], "end_turn"
    else:
        blocks, stop = (
            [{"type": "text", "text": "Scripted reply from " + model}],
            "end_turn",
        )
    msg = {
        "id": "msg_script",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 8},
    }
    if not stream:
        return json.dumps(msg)

    def ev(t, d):
        return f"event: {t}\ndata: {json.dumps(d)}\n\n"

    out = ev(
        "message_start",
        {"type": "message_start", "message": dict(msg, content=[], stop_reason=None)},
    )
    for i, b in enumerate(blocks):
        if b["type"] == "tool_use":
            out += ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": dict(b, input={}),
                },
            )
            out += ev(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(b["input"]),
                    },
                },
            )
        else:
            out += ev(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            if b["text"]:
                out += ev(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": b["text"]},
                    },
                )
        out += ev("content_block_stop", {"type": "content_block_stop", "index": i})
    out += ev(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": 8},
        },
    )
    return out + ev("message_stop", {"type": "message_stop"})


def scripted_openai(parsed, kind, name, args, stream):
    model = parsed.get("model", "gpt")
    message = {"role": "assistant", "content": None}
    finish = "stop"
    if kind == "tool":
        message["tool_calls"] = [
            {
                "id": "call_fp1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
        finish = "tool_calls"
    elif kind == "empty":
        message["content"] = ""
    else:
        message["content"] = "Scripted reply from " + model
    if not stream:
        return json.dumps(
            {
                "id": "chatcmpl-script",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                },
            }
        )
    delta = dict(message)
    if "tool_calls" in delta:
        delta["tool_calls"] = [dict(tc, index=0) for tc in delta["tool_calls"]]

    def chunk(d, f):
        return (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-script",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": model,
                    "choices": [{"index": 0, "delta": d, "finish_reason": f}],
                }
            )
            + "\n\n"
        )

    return chunk(delta, None) + chunk({}, finish) + "data: [DONE]\n\n"


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
        path = self.path.split("?")[0]
        scripted = None
        if SCRIPT_ON and path in ("/v1/messages", "/v1/chat/completions"):
            try:
                parsed = json.loads(raw)
                api = "anthropic" if path == "/v1/messages" else "openai"
                with _lock:  # decide() mutates the cheap-reply counter
                    label, kind, tname, targs = decide(api, parsed)
                entry["scripted"] = label
                scripted = (api, parsed, kind, tname, targs)
            except (ValueError, AttributeError):
                pass
        record(entry)

        if scripted:
            api, parsed, kind, tname, targs = scripted
            build = scripted_anthropic if api == "anthropic" else scripted_openai
            body = build(parsed, kind, tname, targs, is_stream)
            (self.send_sse if is_stream else self.send_json)(200, body)
            return

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
    # The log file is the readiness signal: it exists only once *this* process holds the port.
    if log_path:
        dump_log(log_path)

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
