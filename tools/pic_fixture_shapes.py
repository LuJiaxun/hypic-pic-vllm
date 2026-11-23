# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the prompt fixtures the acceptance runs use.

Four names per shape, and the duplication is deliberate: ``publish``/``reuse`` are the readable pair,
while ``publish_0``/``reuse_0`` are byte-identical copies under the names the shell driver looks for.
Shipping both means a run never has to copy or generate anything.

  a  query <<SEP>> document <<SEP>>
     The document follows a query, so a later request finds it at a position it never occupied and
     the cached KV has to be rotated. Its last token is recomputed because the request samples from it.

  b  system <<SEP>> document <<SEP>> query
     The document keeps its position and only the query after it changes: nothing is rotated. This is
     the shape that should be exact.

  c  system <<SEP>> a <<SEP>> b <<SEP>> c <<SEP>> query
     Three documents published separately, then one request that must reuse the two it has seen and
     recompute the one it has not. The answer is checkable: the question asks about document B and the
     answer is ``dog``.

Documents are prose, not one sentence repeated. A repeated document makes the model continue the
repetition instead of answering, which turns a correct reuse into a wrong answer.

The reused fraction is the point of a timing fixture: with a short document most of the prompt is the
query and nothing is being reused, so the measurement cannot show a gain. Shape b - the one the timings
use - therefore carries a seven thousand token document and a query of about two hundred and seventy,
which leaves the reused fraction slightly under ninety-five percent. Shape a carries four thousand nine
hundred, and the shape that only checks the answer keeps a short document, since reachability is all it
tests.
"""

from __future__ import annotations

SEP = "<<PIC_SEP>>"

SYSTEM = (
    "You are a highly capable and helpful assistant. "
    "Please read the provided documents carefully and answer the question at the end. "
    "Be concise and accurate. If the answer is a single word, output only that word. "
    "Do not add explanations unless explicitly asked. "
) * 4

DOC_A = (
    "The cat is a small domesticated carnivorous mammal. "
    "It is known for its agility, retractable claws, and keen senses. "
    "Cats have been companions to humans for thousands of years. "
    "They are popular pets around the world. "
) * 16

DOC_B = (
    "The dog is a domesticated descendant of the wolf. "
    "It is known for its loyalty, intelligence, and diverse breeds. "
    "Dogs have been working animals and companions to humans for millennia. "
    "They are often called man's best friend. "
) * 16

DOC_C = (
    "The bird is a warm-blooded vertebrate characterized by feathers, wings, and beaks. "
    "It is known for its ability to fly and its diverse species. "
    "Birds are found in nearly every habitat on Earth. "
    "They play important roles in ecosystems. "
) * 16

# The timing shapes reuse most of the prompt, so their document has to be long: at sixteen repetitions
# the reused fraction was under a fifth and the numbers could not be compared with the shape that was
# measured.
DOC_B_LONG = (
    "The dog is a domesticated descendant of the wolf. "
    "It is known for its loyalty, intelligence, and diverse breeds. "
    "Dogs have been working animals and companions to humans for millennia. "
    "They are often called man's best friend. "
) * 110

# Shape b is the one the timings use, and it is longer still: a seven thousand token document with a
# query of about two hundred and seventy. Size is what decides whether reuse pays - the import and the
# matrix apply are a fixed cost, so a document that is too short cannot cover them - and at this length
# four concurrent requests come out about thirty percent ahead of recomputing the same tokens.
DOC_B_TIMED = (
    "The dog is a domesticated descendant of the wolf. "
    "It is known for its loyalty, intelligence, and diverse breeds. "
    "Dogs have been working animals and companions to humans for millennia. "
    "They are often called man's best friend. "
) * 155


# Shape c asks about "segment B", so the documents have to say which segment they are. Without the
# label the model has three documents about animals and no way to tell which one the question means,
# and it answers "which one" instead of "dog". The label is part of the reusable text, so publishing
# and reuse see the same bytes.
DOC_A_SEGMENTED = (
    "segment A is cat. "
    "The cat is a small domesticated carnivorous mammal. "
    "It is known for its agility, retractable claws, and keen senses. "
    "Cats have been companions to humans for thousands of years. "
    "They are popular pets around the world. "
) * 30

DOC_B_SEGMENTED = (
    "segment B is dog. "
    "The dog is a domesticated descendant of the wolf. "
    "It is known for its loyalty, intelligence, and diverse breeds. "
    "Dogs have been working animals and companions to humans for millennia. "
    "They are often called man's best friend. "
) * 30

DOC_C_SEGMENTED = (
    "segment C is bird. "
    "The bird is a warm-blooded vertebrate characterized by feathers, wings, and beaks. "
    "It is known for its ability to fly and its diverse species. "
    "Birds are found in nearly every habitat on Earth. "
    "They play important roles in ecosystems. "
) * 30

QUERY = "Question: which animal is in segment B? Answer with one word."
TARGET = "dog"

# The query is lengthened too, so that the reused fraction is what it is in production rather than
# almost everything. At fourteen tokens the query was noise.
QUERY_LONG = (
    "Please read the document carefully and answer the question at the end. " * 20
) + QUERY

SHAPES = ("a", "b", "c")

#: Names the shell driver publishes before asking for a reuse.
PUBLISH_NAMES = ("publish", "publish_a", "publish_b", "publish_c")


def _with_aliases(files: dict[str, str]) -> dict[str, str]:
    """Add the ``*_0`` copies the shell driver reads.

    A run must not depend on a preparation step: the aliases are written here so the files the driver
    asks for exist after an apply, not after a copy.
    """
    for name in ("publish", "reuse"):
        if name in files:
            files[f"{name}_0"] = files[name]
    if "publish_a" in files:
        files["publish_a_0"] = files["publish_a"]
    return files


def build(shape: str) -> dict[str, str]:
    """Fixture files for one shape, keyed by file name (without ``.txt``)."""
    if shape == "a":
        # The query comes first, so the document sits at a position a later request may not share.
        prompt = f"{QUERY}{SEP}{DOC_B_LONG}{SEP}"
        files = {
            "publish": prompt,
            "reuse": prompt,
            "baseline": f"{QUERY}{DOC_B_LONG}",
        }
    elif shape == "b":
        # The document leads and the query follows, so reuse keeps the document where it was.
        files = {
            "publish": f"{SYSTEM}{SEP}{DOC_B_TIMED}{SEP}",
            "reuse": f"{SYSTEM}{SEP}{DOC_B_TIMED}{SEP}{QUERY_LONG}",
            "baseline": f"{SYSTEM}{DOC_B_TIMED}{QUERY_LONG}",
        }
    elif shape == "c":
        # Three separate publishes, then one request carrying all three documents plus the query.
        files = {
            "publish_a": f"{SYSTEM}{SEP}{DOC_A_SEGMENTED}{SEP}",
            "publish_b": f"{SYSTEM}{SEP}{DOC_B_SEGMENTED}{SEP}",
            "publish_c": f"{SYSTEM}{SEP}{DOC_C_SEGMENTED}{SEP}",
            "reuse": (
                f"{SYSTEM}{SEP}{DOC_A_SEGMENTED}{SEP}{DOC_B_SEGMENTED}{SEP}"
                f"{DOC_C_SEGMENTED}{SEP}{QUERY}"
            ),
            "baseline": f"{SYSTEM}{DOC_A_SEGMENTED}{DOC_B_SEGMENTED}{DOC_C_SEGMENTED}{QUERY}",
        }
    else:
        raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
    return _with_aliases(files)


def segments(text: str) -> list[str]:
    """Every non-empty part of a fixture, in order."""
    return [part for part in text.split(SEP) if part.strip()]


def reusable_parts(text: str) -> list[str]:
    """Only the parts the scheduler may cache.

    Between a pair of separators is reusable; the text outside a pair is a query that differs per
    request. Parts alternate, so the enclosed ones are the odd positions - and only those have to be
    long enough to pass ``--pic-min-segment-tokens``.
    """
    parts = text.split(SEP)
    if len(parts) == 1:
        return [part for part in parts if part.strip()]
    return [
        part for index, part in enumerate(parts) if index % 2 == 1 and part.strip()
    ]


def build_all() -> dict[str, dict[str, str]]:
    return {shape: build(shape) for shape in SHAPES}
