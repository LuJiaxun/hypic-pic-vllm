# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write the acceptance fixtures to disk and report their sizes.

    python3 tools/pic_make_fixtures.py --out /tmp/pic_fixtures
    python3 tools/pic_make_fixtures.py --out /tmp/pic_fixtures --model /path/to/model
    python3 tools/pic_make_fixtures.py --out /tmp/pic_fixtures --shape a     # one shape only

Without ``--model`` the report is in characters. With it, every reusable part is measured in tokens and
anything below ``--min-segment-tokens`` is called out: the scheduler drops such a segment, so a request
that should reuse something reuses nothing and the run looks like a lookup failure rather than a
fixture mistake.

The shapes themselves live in ``tools/pic_fixture_shapes.py`` so the acceptance driver and the tests
read the same definitions.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pic_fixture_shapes import (  # noqa: E402
    SEP,
    SHAPES,
    TARGET,
    build,
    reusable_parts,
    segments,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument(
        "--shape",
        default="all",
        choices=[*SHAPES, "all"],
        help="Which shape to write; the default writes every shape into its own directory.",
    )
    parser.add_argument("--model", default=None, help="Tokenizer path, for token counts.")
    parser.add_argument("--min-segment-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer = None
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    shapes = SHAPES if args.shape == "all" else (args.shape,)
    short: list[tuple[str, str, int]] = []
    written = 0

    for shape in shapes:
        directory = args.out / shape
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in build(shape).items():
            (directory / f"{name}.txt").write_text(text, encoding="utf-8")
            written += 1
            parts = segments(text)
            if tokenizer is None:
                print(f"{shape:7s} {name:12s} chars={len(text):6d} seps={text.count(SEP)}")
                continue
            counts = [
                len(tokenizer.encode(part, add_special_tokens=False)) for part in parts
            ]
            total = sum(counts)
            print(
                f"{shape:7s} {name:12s} segments={len(counts):2d} "
                f"tokens={counts} total={total}"
            )
            # Only the enclosed parts are cached, so only they have to clear the threshold.
            for part in reusable_parts(text):
                count = len(tokenizer.encode(part, add_special_tokens=False))
                if count < args.min_segment_tokens:
                    short.append((shape, name, count))

    print(f"\n{written} file(s) written to {args.out}")
    print(f"expected answer: {TARGET!r}")
    if short:
        print(
            f"\nWARNING: {len(short)} part(s) below --pic-min-segment-tokens "
            f"({args.min_segment_tokens}); the scheduler will skip them:"
        )
        for shape, name, count in short:
            print(f"  {shape}/{name}: {count} tokens")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
