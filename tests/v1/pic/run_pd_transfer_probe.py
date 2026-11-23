# SPDX-License-Identifier: Apache-2.0
"""End-to-end probe for a hybrid Mooncake TransferEngine P/D deployment.

This is deliberately not a "server returned some text" check.  It verifies:
  * a multi-segment request still answers correctly;
  * P actually sent GDN/convolution pages through TransferEngine;
  * D scheduled only P's relayed token, not the original prompt;
  * when a baseline URL is supplied, P/D and the single instance return the
    same token IDs, including the first one.

Stdlib only so it can run next to the servers without extra packages.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

RELAY_RE = re.compile(
    r"PIC-PD relay request=\S+ external_tokens=(\d+) prompt_tokens=(\d+)"
)
STATE_RE = re.compile(
    r"PIC-PD state request=\S+ prompt_tokens=(\d+) pages=(\d+) bytes=(\d+)"
)


# Localhost traffic must never take a corporate/system HTTP proxy: images here
# export http_proxy, and urllib's default opener honours it, which turned a
# 127.0.0.1 request into a 403 HTML page from the proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(
    base_url: str, model: str, prompt: str, max_tokens: int, *, chat: bool = True
) -> tuple[str, list[int]]:
    """Run one request and return ``(text, token_ids)``.

    Shape C uses chat with thinking disabled, which is the semantic acceptance
    path.  ``--completions`` keeps the older raw-completion mode for transport
    debugging.  Both JSON and SSE responses are accepted.
    """
    if chat:
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
                "ignore_eos": True,
                "return_token_ids": True,
            }
        ).encode("utf-8")
        endpoint = "/v1/chat/completions"
    else:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
                "ignore_eos": True,
                "return_token_ids": True,
            }
        ).encode("utf-8")
        endpoint = "/v1/completions"
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _OPENER.open(request, timeout=600.0) as response:
        raw = response.read().decode("utf-8", "replace")

    stripped = raw.lstrip()
    if stripped.startswith("{"):
        # Non-streaming body: either a single instance or a proxy in front of one.
        payload = json.loads(stripped)
        choice = payload["choices"][0]
        if "message" in choice:
            message = choice.get("message") or {}
            text = message.get("content") or message.get("reasoning") or ""
        else:
            text = choice.get("text", "")
        return text, list(choice.get("token_ids") or [])

    text_parts: list[str] = []
    token_ids: list[int] = []
    frames = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        frames += 1
        chunk = json.loads(payload)
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            text = delta.get("content") or delta.get("reasoning")
            if text:
                text_parts.append(text)
            elif choice.get("text"):
                text_parts.append(choice["text"])
            token_ids.extend(choice.get("token_ids") or ())
    if frames == 0:
        raise RuntimeError(
            "P/D proxy returned neither JSON nor SSE data frames: "
            f"{raw[:400]!r}"
        )
    return "".join(text_parts), token_ids


def first_difference(left: list[int], right: list[int]) -> dict | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "baseline": b, "pd": a}
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "baseline": right[index] if index < len(right) else None,
            "pd": left[index] if index < len(left) else None,
        }
    return None


def check_logs(decode_log: Path, prefill_log: Path) -> dict:
    relays = []
    if decode_log.exists():
        for match in RELAY_RE.finditer(decode_log.read_text(encoding="utf-8", errors="replace")):
            relays.append((int(match.group(1)), int(match.group(2))))
    states = []
    if prefill_log.exists():
        for match in STATE_RE.finditer(prefill_log.read_text(encoding="utf-8", errors="replace")):
            states.append(
                (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            )
    relay_ok = bool(relays) and all(prompt == external + 1 for external, prompt in relays)
    state_ok = bool(states) and all(pages > 0 and nbytes > 0 for _, pages, nbytes in states)
    boundary_ok = False
    if relays and states:
        external = relays[-1][0]
        boundary_ok = any(prompt == external for prompt, _, _ in states)
    return {
        "relay_events": len(relays),
        "state_events": len(states),
        "relay_only_one_token": relay_ok,
        "gdn_state_transferred": state_ok,
        "same_prompt_boundary": boundary_ok,
        "last_external_tokens": relays[-1][0] if relays else None,
        "last_state_pages": states[-1][1] if states else None,
        "last_state_bytes": states[-1][2] if states else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--baseline-url", default=None)
    parser.add_argument("--model", default="qwen3.5-2b")
    parser.add_argument(
        "--prompt-file",
        default="tests/v1/pic/fixtures/c/reuse.txt",
    )
    parser.add_argument("--expect", default="dog")
    parser.add_argument(
        "--completions",
        action="store_true",
        help="Use raw /v1/completions instead of the default semantic chat path",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--decode-log", default="/tmp/pic_pd_decode.log")
    parser.add_argument("--prefill-log", default="/tmp/pic_pd_prefill.log")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    pd_answers: list[tuple[str, list[int]]] = []
    for index in range(args.repeat):
        text, token_ids = post(
            args.base_url,
            args.model,
            prompt,
            args.max_tokens,
            chat=not args.completions,
        )
        print(f"P/D run {index + 1}: tokens={len(token_ids)} answer={text!r}")
        if not token_ids:
            raise SystemExit("FAIL: P/D returned no token IDs")
        if args.expect.lower() not in text.lower():
            raise SystemExit(f"FAIL: expected {args.expect!r} in {text!r}")
        pd_answers.append((text, token_ids))

    deterministic = all(tokens == pd_answers[0][1] for _, tokens in pd_answers[1:])
    if not deterministic:
        raise SystemExit("FAIL: repeated P/D runs returned different token IDs")

    baseline_equal = None
    if args.baseline_url:
        baseline_text, baseline_tokens = post(
            args.baseline_url,
            args.model,
            prompt,
            args.max_tokens,
            chat=not args.completions,
        )
        baseline_equal = baseline_tokens == pd_answers[0][1]
        print(f"baseline: tokens={len(baseline_tokens)} answer={baseline_text!r}")
        if not baseline_equal:
            difference = first_difference(pd_answers[0][1], baseline_tokens)
            raise SystemExit(f"FAIL: P/D differs from baseline at {difference}")

    logs = check_logs(Path(args.decode_log), Path(args.prefill_log))
    print("transport:", json.dumps(logs, ensure_ascii=False))
    for key in ("relay_only_one_token", "gdn_state_transferred", "same_prompt_boundary"):
        if not logs[key]:
            raise SystemExit(f"FAIL: transport check {key} is false")

    result = {
        "status": "PASS",
        "answer": pd_answers[0][0],
        "token_count": len(pd_answers[0][1]),
        "repeat_deterministic": deterministic,
        "baseline_token_ids_equal": baseline_equal,
        **logs,
    }
    print("PASS:", json.dumps(result, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500]
        print(f"FAIL HTTP {exc.code}: {detail!r}", file=sys.stderr)
        print(
            "Check the proxy log for a traceback; the body above is what the "
            "proxy returned (JSON error vs HTML/text points at different causes).",
            file=sys.stderr,
        )
        sys.exit(1)
