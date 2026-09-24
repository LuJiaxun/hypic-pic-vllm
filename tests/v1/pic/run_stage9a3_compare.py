#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the Stage 9-A-3 reference-vs-PIC completion comparison.

The vLLM server must already be running.  This script performs one single-
segment warmup, then sends the same three-segment prompt once without PIC and
once with PIC.  Runtime-path requirements are checked from the server log
separately, as described in the Stage 9-A-3 README.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.request import Request, urlopen

from vllm.v1.pic.validation import compare_completion_responses


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("server response is not a JSON object")
    return body


def _tokenize(base_url: str, model: str, text: str) -> list[int]:
    response = _post(
        f"{base_url.rstrip('/')}/tokenize",
        {
            "model": model,
            "prompt": text,
            "add_special_tokens": False,
        },
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValueError("/tokenize did not return an integer token list")
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-2b")
    parser.add_argument("--left", default="X1")
    parser.add_argument("--target", required=True)
    parser.add_argument("--right", default="Y1")
    parser.add_argument("--separator", default="<<PIC_SEP>>")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seam-sink", type=int, default=4)
    args = parser.parse_args()

    common = {
        "model": args.model,
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
        "max_tokens": args.max_tokens,
        "logprobs": 5,
    }
    warmup = dict(common)
    warmup["prompt"] = args.target
    warmup["vllm_xargs"] = {
        "pic_enabled": True,
        "pic_auto_enable": True,
        "pic_separator": args.separator,
        "pic_seam_sink": 0,
    }
    _post(f"{args.base_url.rstrip('/')}/v1/completions", warmup)

    prompt = f"{args.left}{args.separator}{args.target}{args.separator}{args.right}"
    # PIC tokenizes each segment independently.  Build the reference request
    # from the server's own tokenizer so BPE boundary merges and tokenizer
    # versions cannot make the two requests use different input IDs.
    pic_token_ids = [
        token
        for part in (args.left, args.target, args.right)
        for token in _tokenize(args.base_url, args.model, part)
    ]
    reference = dict(common)
    reference["prompt"] = pic_token_ids
    reference["vllm_xargs"] = {"pic_enabled": False}
    reference_response = _post(
        f"{args.base_url.rstrip('/')}/v1/completions", reference
    )

    pic = dict(common)
    pic["prompt"] = prompt
    pic["vllm_xargs"] = {
        "pic_enabled": True,
        "pic_auto_enable": True,
        "pic_separator": args.separator,
        "pic_mode": "transition_rope_recompute",
        "pic_seam_sink": args.seam_sink,
    }
    pic_response = _post(f"{args.base_url.rstrip('/')}/v1/completions", pic)

    result = compare_completion_responses(reference_response, pic_response)
    print(
        json.dumps(
            {
                "passed": result.passed,
                "text_equal": result.text_equal,
                "prompt_tokens_equal": result.prompt_tokens_equal,
                "completion_tokens_equal": result.completion_tokens_equal,
                "logprobs_compared": result.logprobs_compared,
                "logprobs_equal": result.logprobs_equal,
                "max_logprob_abs_diff": result.max_logprob_abs_diff,
                "reference_text": result.reference_text,
                "pic_text": result.pic_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
