# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure a set of distinct reusable prompts at several concurrency levels.

The per-prompt probe answers "how fast is one document when it is reused N times". Real traffic is
the other shape: many different documents, each arriving once. That changes what is being measured -
every request in flight carries its own document, so imports never share a transfer and the
concurrency level is a property of the set rather than of one prompt.

Each round is therefore one request per prompt:

    pre-publish all prompts  ->  round(c=1)  ->  round(c=4)

The publisher is warmed first so both rounds meet the same already-published state; otherwise the
first round would pay publishing costs the second one does not, and the two levels would not be
comparable.

    python3 tests/v1/pic/run_pic_prompt_set_probe.py \
        --base-url http://127.0.0.1:8001 --model qwen3.5-2b \
        --publish-dir /path/publish --reuse-dir /path/reuse \
        --concurrency 1,4 --json-out /tmp/prompt_set.json
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


def post_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> tuple[bool, float, str]:
    """Return ``(ok, elapsed_ms, detail)`` for one completion request."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body, headers=headers
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        elapsed = (time.perf_counter() - started) * 1000.0
        if not payload:
            return False, elapsed, "empty response"
        return True, elapsed, ""
    except urllib.error.HTTPError as exc:
        return False, 0.0, f"HTTP {exc.code}: {exc.read()[:200]!r}"
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return False, 0.0, f"{type(exc).__name__}: {exc}"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def load_prompts(directory: Path, prefix: str) -> list[tuple[str, str]]:
    """Return ``(name, text)`` for ``<prefix><n>.txt`` in numeric order."""
    found: list[tuple[int, str, str]] = []
    for path in directory.glob(f"{prefix}*.txt"):
        stem = path.stem[len(prefix) :]
        if not stem.isdigit():
            continue
        found.append((int(stem), path.name, path.read_text(encoding="utf-8")))
    found.sort()
    return [(name, text) for _, name, text in found]


def summarize(
    label: str, latencies: list[float], wall_ms: float, failures: list[str] | None = None
) -> dict:
    result = {
        "stage": label,
        "requests": len(latencies),
        "wall_ms": round(wall_ms, 3),
        "p50_ms": round(percentile(latencies, 0.50) or 0.0, 3) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95) or 0.0, 3) if latencies else None,
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else None,
        "min_ms": round(min(latencies), 3) if latencies else None,
        "max_ms": round(max(latencies), 3) if latencies else None,
        "throughput_rps": (
            round(len(latencies) / (wall_ms / 1000.0), 4) if wall_ms > 0 and latencies else None
        ),
        "failed": len(failures or ()),
        "failures": (failures or [])[:5],
    }
    print(json.dumps(result, ensure_ascii=False))
    return result


def publish_all(args: argparse.Namespace, prompts: list[tuple[str, str]]) -> dict:
    """Publish every prompt once, serially, and report the cost."""
    latencies: list[float] = []
    failures: list[str] = []
    started = time.perf_counter()
    for name, text in prompts:
        ok, elapsed, detail = post_completion(
            base_url=args.publish_base_url or args.base_url,
            model=args.model,
            prompt=text,
            max_tokens=1,
            timeout=args.timeout,
            api_key=args.api_key,
        )
        if ok:
            latencies.append(elapsed)
        else:
            failures.append(f"{name}: {detail}")
    wall_ms = (time.perf_counter() - started) * 1000.0
    return summarize("publish", latencies, wall_ms, failures)


def run_round(args: argparse.Namespace, prompts: list[tuple[str, str]], level: int) -> dict:
    """Send one request per prompt at the given concurrency."""
    for _ in range(args.warmup):
        ok, _, detail = post_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=prompts[0][1],
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            api_key=args.api_key,
        )
        if not ok:
            raise SystemExit(f"warmup request failed: {detail}")

    latencies: list[float] = []
    failures: list[str] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, level)) as pool:
        futures = [
            (
                name,
                pool.submit(
                    post_completion,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=text,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    api_key=args.api_key,
                ),
            )
            for name, text in prompts
        ]
        for name, future in futures:
            ok, elapsed, detail = future.result()
            if ok:
                latencies.append(elapsed)
            else:
                failures.append(f"{name}: {detail}")
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    result = summarize(f"reuse_c{level}", latencies, wall_ms, failures)
    result["concurrency"] = level
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--publish-base-url",
        default=None,
        help="where prompts are published; defaults to --base-url",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--publish-dir", required=True)
    parser.add_argument("--reuse-dir", required=True)
    parser.add_argument("--publish-prefix", default="publish_")
    parser.add_argument("--reuse-prefix", default="reuse_")
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="the prompts are already published; only run the reuse rounds",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    publish_prompts = load_prompts(Path(args.publish_dir), args.publish_prefix)
    reuse_prompts = load_prompts(Path(args.reuse_dir), args.reuse_prefix)
    if not reuse_prompts:
        print(f"no reuse prompts matched {args.reuse_prefix}*.txt in {args.reuse_dir}")
        return 2
    if publish_prompts and len(publish_prompts) != len(reuse_prompts):
        print(
            f"publish/reuse counts differ: {len(publish_prompts)} vs {len(reuse_prompts)}; "
            "pairing by index may be wrong"
        )

    print(
        json.dumps(
            {
                "stage": "setup",
                "publish_prompts": len(publish_prompts),
                "reuse_prompts": len(reuse_prompts),
                "concurrency": args.concurrency,
            }
        )
    )

    publish_result = None
    if not args.skip_publish and publish_prompts:
        publish_result = publish_all(args, publish_prompts)

    levels = sorted(int(item) for item in str(args.concurrency).split(",") if item.strip())
    rounds = [run_round(args, reuse_prompts, level) for level in levels]

    print("\nstage | p50_ms | p95_ms | tput_rps | failed")
    if publish_result is not None:
        print(
            f"publish | {publish_result['p50_ms']} | {publish_result['p95_ms']} | "
            f"{publish_result['throughput_rps']} | {publish_result['failed']}"
        )
    for item in rounds:
        print(
            f"c={item['concurrency']} | {item['p50_ms']} | {item['p95_ms']} | "
            f"{item['throughput_rps']} | {item['failed']}"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "prompt_count": len(reuse_prompts),
                    "warmup": args.warmup,
                    "publish": publish_result,
                    "rounds": rounds,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
