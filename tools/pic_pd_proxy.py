# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated P/D proxy for hybrid (attention + GDN/Mamba) models.

This is the proxy the P/D test launcher uses.  It exists here rather than under
``examples/`` because a pip-installed vLLM ships no ``examples/`` directory, so
a patch that merely edits the upstream example cannot be applied to a
``dist-packages`` tree at all.

``--relay-first-token`` implements the hybrid handoff:

* the prefiller runs the original prompt and returns its first sampled token;
* the decoder receives ``prompt_token_ids + [first_token]`` and only forwards
  that token, starting from the prefiller's live GDN state;
* the first token is also emitted as the first output chunk, because the
  decoder consumes it as input and would otherwise swallow it.

Without the flag the proxy behaves like the upstream example: the prefiller
runs in the background and the decoder streams the whole answer.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import itertools
import json
import os
import urllib
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


def make_http_path(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def prefiller_cycle(prefill_clients: list[Any]):
    while True:
        for prefill_client in prefill_clients:
            for i in range(prefill_client["dp_size"]):
                yield prefill_client, i


async def get_prefiller_info(prefill_clients: list, ready: asyncio.Event):
    for prefill_client in prefill_clients:
        while True:
            try:
                response = await prefill_client["client"].get("/health")
                response.raise_for_status()
            except Exception:
                await asyncio.sleep(1)
                continue

            response = await prefill_client["client"].get(
                prefill_client["bootstrap_addr"] + "/query"
            )
            response.raise_for_status()
            data = response.json()
            break

        for dp_rank, dp_entry in data.items():
            prefill_client["dp_engine_id"][int(dp_rank)] = dp_entry["engine_id"]
        dp_size = len(data)
        prefill_client["dp_size"] = dp_size
        print(f"Inited prefiller {prefill_client['url']} with dp_size={dp_size}")

    ready.set()
    print("All prefiller instances are ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.ready = asyncio.Event()

    for url, bootstrap_port in global_args.prefill:
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "url": url,
                "bootstrap_addr": make_http_path(hostname, bootstrap_port or 8998),
                "dp_engine_id": {},
            }
        )

    for url in global_args.decode:
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
            }
        )

    asyncio.create_task(get_prefiller_info(app.state.prefill_clients, app.state.ready))

    app.state.prefill_iterator = prefiller_cycle(app.state.prefill_clients)
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Got {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()
    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument(
        "--relay-first-token",
        action="store_true",
        help=(
            "Relay the prefiller's first sampled token to the decoder at "
            "token-ID precision. Required for GDN/Mamba P/D: the decoder "
            "forwards that token from the prefiller's live h_N state instead "
            "of advancing the original prompt's last token twice."
        ),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "bootstrap_port"),
        help=(
            "Prefill server URL and optional bootstrap port. "
            "Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, 'none', or omitted."
        ),
    )
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_raw",
        metavar=("URL",),
        help="Decode server URL. Can be specified multiple times.",
    )

    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_raw)
    args.decode = _parse_decode_urls(args.decode_raw)
    return args


# From sglang router_args.py
def _parse_prefill_urls(prefill_list):
    """Parse ``--prefill URL [BOOTSTRAP_PORT]`` entries."""
    if not prefill_list:
        return []

    prefill_urls = []
    for prefill_args in prefill_list:
        url = prefill_args[0]

        if len(prefill_args) >= 2:
            bootstrap_port_str = prefill_args[1]
            if bootstrap_port_str.lower() == "none":
                bootstrap_port = None
            else:
                try:
                    bootstrap_port = int(bootstrap_port_str)
                except ValueError as e:
                    raise ValueError(
                        "Invalid bootstrap port: "
                        f"{bootstrap_port_str}. Must be a number or 'none'"
                    ) from e
        else:
            bootstrap_port = None

        prefill_urls.append((url, bootstrap_port))

    return prefill_urls


def _parse_decode_urls(decode_list):
    """Parse ``--decode URL`` entries."""
    if not decode_list:
        return []
    return [url[0] for url in decode_list]


def get_next_client(app, service_type: str):
    if service_type == "prefill":
        return next(app.state.prefill_iterator)
    if service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, dp_rank: int, endpoint: str, req_data: dict, request_id: str
) -> tuple[list[int], int, dict, dict] | None:
    """Run the prefiller for one token, optionally returning exact token IDs."""
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": f"xfer-{request_id}",
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    if global_args.relay_first_token:
        req_data["return_token_ids"] = True
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(dp_rank),
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    payload = response.json()
    # CRITICAL: Release connection back to pool
    await response.aclose()
    if not global_args.relay_first_token:
        return None

    choice = payload["choices"][0]
    prompt_token_ids = payload.get("prompt_token_ids") or choice.get("prompt_token_ids")
    sampled_token_ids = choice.get("token_ids")
    if not prompt_token_ids or not sampled_token_ids:
        raise RuntimeError(
            "P/D first-token relay requires return_token_ids on the prefiller"
        )
    if len(sampled_token_ids) != 1:
        raise RuntimeError("P/D relay expected exactly one sampled token")
    return list(prompt_token_ids), int(sampled_token_ids[0]), choice, payload


def _relay_prefix_chunk(endpoint: str, relay: tuple) -> str:
    """Server-sent event carrying the prefiller's first token."""
    _, first_token, choice, payload = relay
    if endpoint == "/v1/completions":
        chunk = {
            "id": payload.get("id"),
            "object": "text_completion",
            "created": payload.get("created"),
            "model": payload.get("model"),
            "choices": [
                {
                    "index": 0,
                    "text": choice.get("text", ""),
                    "logprobs": None,
                    "finish_reason": None,
                    "token_ids": [first_token],
                }
            ],
        }
    else:
        message = choice.get("message") or {}
        delta: dict[str, Any] = {"role": "assistant"}
        for field in ("content", "reasoning", "tool_calls"):
            if message.get(field) is not None:
                delta[field] = message[field]
        chunk = {
            "id": payload.get("id"),
            "object": "chat.completion.chunk",
            "created": payload.get("created"),
            "model": payload.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": None,
                    "token_ids": [first_token],
                }
            ],
        }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def stream_service_response(
    prefill_client_info: dict,
    prefill_dp_rank: int,
    decode_client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    relay: tuple | None = None,
):
    """Stream the decoder response, prefixed by the relayed token when enabled."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    req_data["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
    }

    if relay is not None:
        prompt_token_ids = relay[0]
        first_token = relay[1]
        if endpoint == "/v1/completions":
            req_data["prompt"] = [*prompt_token_ids, first_token]
        else:
            # Chat requests use the token-exact override; never detokenize and
            # rerun the chat template or the GDN boundary moves.
            req_data["prompt_token_ids"] = [*prompt_token_ids, first_token]
        # Only the original prompt has KV on the prefiller.  The decoder
        # forwards the relayed token itself from the prefiller's live state.
        req_data["kv_transfer_params"]["remote_prefill_tokens"] = len(
            prompt_token_ids
        )
        yield _relay_prefix_chunk(endpoint, relay)

    async with decode_client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def request_service_response(
    prefill_client_info: dict,
    prefill_dp_rank: int,
    decode_client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    relay: tuple | None = None,
) -> dict:
    """Return one JSON completion, including the relayed first token.

    The streaming path below is correct for streaming clients, but a client that
    explicitly asks for ``stream=false`` must not receive SSE text.  Here the
    decoder returns ordinary JSON and the prefiller's first token is merged back
    into it, preserving both text and token IDs.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
    }
    if relay is not None:
        prompt_token_ids = relay[0]
        first_token = relay[1]
        if endpoint == "/v1/completions":
            req_data["prompt"] = [*prompt_token_ids, first_token]
        else:
            req_data["prompt_token_ids"] = [*prompt_token_ids, first_token]
        req_data["kv_transfer_params"]["remote_prefill_tokens"] = len(
            prompt_token_ids
        )
    req_data["stream"] = False
    response = await decode_client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()
    payload = response.json()
    await response.aclose()

    if relay is None:
        return payload
    prompt_token_ids, first_token, p_choice, p_payload = relay
    d_choice = payload["choices"][0]
    if endpoint == "/v1/completions":
        d_choice["text"] = p_choice.get("text", "") + d_choice.get("text", "")
        d_choice["prompt_token_ids"] = prompt_token_ids
    else:
        p_message = p_choice.get("message") or {}
        d_message = d_choice.get("message") or {}
        for field in ("content", "reasoning"):
            if p_message.get(field) is not None or d_message.get(field) is not None:
                d_message[field] = (p_message.get(field) or "") + (
                    d_message.get(field) or ""
                )
        payload["prompt_token_ids"] = prompt_token_ids
    d_choice["token_ids"] = [first_token] + list(d_choice.get("token_ids") or [])

    usage = payload.get("usage")
    if isinstance(usage, dict):
        p_usage = p_payload.get("usage") or {}
        if p_usage.get("prompt_tokens") is not None:
            usage["prompt_tokens"] = p_usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            usage["completion_tokens"] = int(usage["completion_tokens"]) + 1
        if usage.get("prompt_tokens") is not None:
            usage["total_tokens"] = int(usage["prompt_tokens"]) + int(
                usage.get("completion_tokens", 0)
            )
    return payload


async def _handle_completions(api: str, request: Request):
    if not app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")

    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        prefill_client_info, prefill_dp_rank = get_next_client(request.app, "prefill")

        if global_args.relay_first_token:
            relay = await send_request_to_service(
                prefill_client_info, prefill_dp_rank, api, req_data, request_id
            )
        else:
            relay = None
            asyncio.create_task(
                send_request_to_service(
                    prefill_client_info, prefill_dp_rank, api, req_data, request_id
                )
            )

        decode_client_info = get_next_client(request.app, "decode")

        if req_data.get("stream", True) is False:
            return await request_service_response(
                prefill_client_info,
                prefill_dp_rank,
                decode_client_info,
                api,
                req_data,
                request_id=request_id,
                relay=relay,
            )

        async def generate_stream():
            async for chunk in stream_service_response(
                prefill_client_info,
                prefill_dp_rank,
                decode_client_info,
                api,
                req_data,
                request_id=request_id,
                relay=relay,
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.get("/health")
async def health():
    """Readiness probe used by tools/test_pd_transfer_engine.sh."""
    return {"status": "ok", "ready": bool(app.state.ready.is_set())}


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


if __name__ == "__main__":
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
