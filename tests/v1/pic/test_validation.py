# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.pic.validation import compare_completion_responses


def _response(text: str, *, prompt: int = 10, completion: int = 3):
    return {
        "choices": [{"text": text, "logprobs": {"token_logprobs": [-0.1, -0.2]}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        },
    }


def test_response_validation_accepts_equal_text_usage_and_logprobs() -> None:
    result = compare_completion_responses(_response("answer"), _response("answer"))

    assert result.passed
    assert result.logprobs_compared
    assert result.max_logprob_abs_diff == 0.0


def test_response_validation_rejects_different_generated_text() -> None:
    result = compare_completion_responses(_response("reference"), _response("pic"))

    assert not result.passed
    assert not result.text_equal


def test_response_validation_uses_tolerance_for_logprobs() -> None:
    reference = _response("answer")
    pic = _response("answer")
    pic["choices"][0]["logprobs"]["token_logprobs"] = [-0.1005, -0.1995]

    result = compare_completion_responses(reference, pic)

    assert result.passed
    assert result.max_logprob_abs_diff == pytest.approx(0.0005)
