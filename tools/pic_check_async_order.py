"""Verify the ordering invariants of the PIC parked-import path from a log.

The import for a parked request is started on one step, noticed on a later step, and
adopted when the request is finally scheduled.  A mistake in that ordering either loses
the overlap or lets attention read blocks that were never imported, and neither shows up
in a latency number.

Rules checked per request id (all step-based, no wall clock):

    SUBMIT appears once
    READY or FAILED appears after SUBMIT
    ADOPT appears exactly once and not before READY
    the request's first "forward ... request=<rid>" is at a step later than ADOPT

Usage:  python tools/pic_check_async_order.py <log> [--strict]
Non-zero exit code lists every violation.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

SUBMIT = re.compile(r"async step=(?P<step>\d+) SUBMIT request=(?P<rid>\S+)")
READY = re.compile(r"async step=(?P<step>\d+) READY request=(?P<rid>\S+)")
FAILED = re.compile(r"async step=(?P<step>\d+) FAILED request=(?P<rid>\S+)")
ADOPT = re.compile(r"async step=(?P<step>\d+) ADOPT request=(?P<rid>\S+)")
FORWARD = re.compile(r"forward request=(?P<rid>[^\s,]+)")
STEP = re.compile(r"step=(?P<step>\d+)")


def check(path: str, strict: bool) -> int:
    submits: dict[str, int] = {}
    readies: dict[str, int] = {}
    failures: dict[str, int] = {}
    adopts: dict[str, list[int]] = {}
    forwards: dict[str, int] = {}
    duplicated: set[str] = set()

    current = 0
    for line in pathlib.Path(path).read_text(errors="replace").splitlines():
        if m := STEP.search(line):
            current = int(m.group("step"))
        if m := SUBMIT.search(line):
            rid = m.group("rid")
            if rid in submits:
                duplicated.add(rid)
            submits[rid] = int(m.group("step"))
        elif m := READY.search(line):
            readies[m.group("rid")] = int(m.group("step"))
        elif m := FAILED.search(line):
            failures[m.group("rid")] = int(m.group("step"))
        elif m := ADOPT.search(line):
            adopts.setdefault(m.group("rid"), []).append(int(m.group("step")))
        elif m := FORWARD.search(line):
            forwards.setdefault(m.group("rid"), current)

    problems: list[str] = []
    for rid in duplicated:
        problems.append(f"{rid}: submitted more than once")
    for rid, submit_step in submits.items():
        done = readies.get(rid) or failures.get(rid)
        if done is None:
            if strict:
                problems.append(f"{rid}: submitted at step {submit_step} but never finished")
            continue
        if done < submit_step:
            problems.append(
                f"{rid}: finished at step {done}, before its submit step {submit_step}"
            )
    for rid, steps in adopts.items():
        if len(steps) != 1:
            problems.append(f"{rid}: adopted {len(steps)} times, expected 1")
        ready = readies.get(rid)
        if ready is not None and steps[0] < ready:
            problems.append(f"{rid}: adopted at step {steps[0]} before ready step {ready}")
        forward = forwards.get(rid)
        if forward is not None and forward < steps[0]:
            problems.append(
                f"{rid}: forwarded at step {forward}, before its adopt step {steps[0]}"
            )
    for rid in readies:
        if rid not in adopts and strict:
            problems.append(f"{rid}: became ready but was never adopted")

    if problems:
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} ordering violation(s)")
        return 1
    print(
        f"clean: {len(submits)} submit(s), {len(readies)} ready, "
        f"{len(failures)} failed, {len(adopts)} adopt"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    return check(args.log, args.strict)


if __name__ == "__main__":
    sys.exit(main())
