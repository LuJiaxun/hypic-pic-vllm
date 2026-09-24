# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-vs-PIC response validation helpers for Stage 9-A-3.

This module intentionally stays outside the execution path.  It compares a
normal reference completion with a PIC completion after the server has run the
same prompt.  The worker runtime trace remains the authority for skipped ranges
and transition execution; this module only evaluates observable responses and
returned generation logprobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PICResponseValidation:
    """Response-level result for one reference/PIC completion pair."""

    reference_text: str
    pic_text: str
    text_equal: bool
    prompt_tokens_equal: bool
    completion_tokens_equal: bool
    logprobs_compared: bool
    logprobs_equal: bool
    max_logprob_abs_diff: float | None

    @property
    def passed(self) -> bool:
        return (
            self.text_equal
            and self.prompt_tokens_equal
            and self.completion_tokens_equal
            and (not self.logprobs_compared or self.logprobs_equal)
        )


def _choice_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("completion response has no choices")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise ValueError("completion response choice has no text")
    return text


def _usage_value(response: Mapping[str, Any], key: str) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def _token_logprobs(response: Mapping[str, Any]) -> tuple[float, ...] | None:
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return None
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, Mapping):
        return None
    values = logprobs.get("token_logprobs")
    if not isinstance(values, Sequence):
        return None
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        result.append(float(value))
    return tuple(result)


def compare_completion_responses(
    reference: Mapping[str, Any],
    pic: Mapping[str, Any],
    *,
    logprob_atol: float = 1e-3,
    logprob_rtol: float = 1e-3,
) -> PICResponseValidation:
    """Compare deterministic reference and PIC completion responses.

    Missing usage fields are treated as uncomparable rather than as a false
    mismatch.  When both responses contain token logprobs, all returned values
    must have equal length and satisfy the supplied tolerance.
    """
    reference_logprobs = _token_logprobs(reference)
    pic_logprobs = _token_logprobs(pic)
    logprobs_compared = reference_logprobs is not None and pic_logprobs is not None
    max_diff: float | None = None
    logprobs_equal = True
    if logprobs_compared:
        assert reference_logprobs is not None
        assert pic_logprobs is not None
        if len(reference_logprobs) != len(pic_logprobs):
            logprobs_equal = False
        else:
            diffs = [
                abs(reference_value - pic_value)
                for reference_value, pic_value in zip(
                    reference_logprobs, pic_logprobs
                )
            ]
            max_diff = max(diffs, default=0.0)
            logprobs_equal = all(
                isclose(
                    reference_value,
                    pic_value,
                    rel_tol=logprob_rtol,
                    abs_tol=logprob_atol,
                )
                for reference_value, pic_value in zip(
                    reference_logprobs, pic_logprobs
                )
            )

    reference_prompt_tokens = _usage_value(reference, "prompt_tokens")
    pic_prompt_tokens = _usage_value(pic, "prompt_tokens")
    reference_completion_tokens = _usage_value(reference, "completion_tokens")
    pic_completion_tokens = _usage_value(pic, "completion_tokens")

    return PICResponseValidation(
        reference_text=_choice_text(reference),
        pic_text=_choice_text(pic),
        text_equal=_choice_text(reference) == _choice_text(pic),
        prompt_tokens_equal=(
            reference_prompt_tokens is None
            or pic_prompt_tokens is None
            or reference_prompt_tokens == pic_prompt_tokens
        ),
        completion_tokens_equal=(
            reference_completion_tokens is None
            or pic_completion_tokens is None
            or reference_completion_tokens == pic_completion_tokens
        ),
        logprobs_compared=logprobs_compared,
        logprobs_equal=logprobs_equal,
        max_logprob_abs_diff=max_diff,
    )
