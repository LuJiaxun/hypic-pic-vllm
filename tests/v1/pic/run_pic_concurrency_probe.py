# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Concurrency probe for cross-process PIC reuse.

The Stage 11E-B report measured a *sequential* request loop. The open question
is what happens when several PIC requests are in flight at once, because the
worker imports each new remote request inline in ``_update_states()``: request
N+1's import starts only after request N's import returned. This probe measures
that directly.

Usage (server side, stdlib only):

    python3 tests/v1/pic/run_pic_concurrency_probe.py \
        --base-url http://127.0.0.1:8000 --model qwen3.5-2b \
        --publish-prompt-file /tmp/publish.txt \
        --reuse-prompt-file /tmp/reuse.txt \
        --concurrency 1,2,4 --requests 20 --warmup 3 \
        --json-out /tmp/pic_concurrency.json

It sends OpenAI ``/v1/completions`` requests, so any vLLM endpoint works. Run it
once per concurrency level while ``HYPIC_PIC_DEBUG=1`` is set on the
server, then correlate with ``tools/pic_import_profile.py``:

    python3 tools/pic_import_profile.py /tmp/pic_b.log --json > /tmp/pic_b.json

Reading the result:

- ``pic_p50`` roughly constant across levels -> imports overlap;
- ``pic_p50`` grows with the level -> imports are serialised (N x import cost),
  which is exactly what the B3 work targets.

Do not use this probe as a correctness check. It only measures latency and
success counts; output equality is covered by the Stage 9-A-3 scripts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def build_prompt(words: int, tag: str) -> str:
    """Deterministic filler prompt; word count is not a token count."""

    return " ".join(f"{tag}{index}" for index in range(max(0, words)))


def compose(prefix: str, reuse: str, suffix: str, separator: str) -> str:
    return f"{prefix}{separator}{reuse}{separator}{suffix}"


def read_text(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def post_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> tuple[bool, float, str]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "ignore_eos": True,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        elapsed = (time.perf_counter() - start) * 1000.0
        if response.status != 200:
            return False, elapsed, f"HTTP {response.status}"
        return True, elapsed, body.decode("utf-8", "replace")[:200]
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        return False, elapsed, f"HTTP {exc.code}: {exc.read()[:200]!r}"
    except Exception as exc:  # noqa: BLE001 - reported as a probe failure
        elapsed = (time.perf_counter() - start) * 1000.0
        return False, elapsed, repr(exc)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def run_level(
    *,
    args: argparse.Namespace,
    prompt: str,
    concurrency: int,
) -> dict[str, object]:
    for _ in range(args.warmup):
        ok, _, detail = post_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            api_key=args.api_key,
        )
        if not ok:
            raise SystemExit(f"warmup request failed: {detail}")

    latencies: list[float] = []
    failures: list[str] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                post_completion,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                api_key=args.api_key,
            )
            for _ in range(args.requests)
        ]
        for future in futures:
            ok, elapsed, detail = future.result()
            if ok:
                latencies.append(elapsed)
            else:
                failures.append(detail)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    result = {
        "concurrency": concurrency,
        "requests": args.requests,
        "success": len(latencies),
        "failed": len(failures),
        "wall_ms": round(wall_ms, 3),
        "p50_ms": round(percentile(latencies, 0.50), 3) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95), 3) if latencies else None,
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        "min_ms": round(min(latencies), 3) if latencies else None,
        "max_ms": round(max(latencies), 3) if latencies else None,
        "throughput_rps": (
            round(len(latencies) / (wall_ms / 1000.0), 4) if wall_ms > 0 else None
        ),
        "failures": failures[:5],
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--separator", default="<<PIC_SEP>>")
    parser.add_argument("--publish-prompt-file", default=None)
    parser.add_argument("--reuse-prompt-file", default=None)
    parser.add_argument(
        "--reuse-words",
        type=int,
        default=3200,
        help="filler words for the middle (reused) segment when no prompt file",
    )
    parser.add_argument("--side-words", type=int, default=256)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    publish_prompt = read_text(args.publish_prompt_file)
    if publish_prompt is None:
        reuse_segment = build_prompt(args.reuse_words, "b")
        publish_prompt = compose(
            build_prompt(args.side_words, "a"), reuse_segment,
            build_prompt(args.side_words, "c"), args.separator,
        )
    reuse_prompt = read_text(args.reuse_prompt_file)
    if reuse_prompt is None:
        reuse_segment = build_prompt(args.reuse_words, "b")
        reuse_prompt = compose(
            build_prompt(args.side_words, "x"), reuse_segment,
            build_prompt(args.side_words, "y"), args.separator,
        )

    print(
        json.dumps(
            {
                "stage": "publish",
                "prompt_chars": len(publish_prompt),
                "max_tokens": args.max_tokens,
            }
        )
    )
    ok, elapsed, detail = post_completion(
        base_url=args.base_url,
        model=args.model,
        prompt=publish_prompt,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    if not ok:
        raise SystemExit(f"publish request failed: {detail}")
    print(json.dumps({"stage": "publish", "elapsed_ms": round(elapsed, 3)}))

    levels = [int(item) for item in str(args.concurrency).split(",") if item.strip()]
    results = [
        run_level(args=args, prompt=reuse_prompt, concurrency=level)
        for level in sorted(levels)
    ]

    print("\nconcurrency | p50_ms | p95_ms | tput_rps | failed")
    for item in results:
        print(
            f"{item['concurrency']:>11} | {item['p50_ms']} | {item['p95_ms']} | "
            f"{item['throughput_rps']} | {item['failed']}"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "requests_per_level": args.requests,
                    "warmup": args.warmup,
                    "levels": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")

    print(
        "\nNext: python3 tools/pic_import_profile.py /tmp/pic_b.log --json\n"
        "and compare native-import / transition-transfer sums against the\n"
        "level counts. Constant p50 means imports overlap; a linear p50 growth\n"
        "means they are serialised inside _update_states()."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
