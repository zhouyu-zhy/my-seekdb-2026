#!/usr/bin/env python3
"""Validate `agent/conversation.json` against the checks the judge performs.

Implements the mechanical parts of section 3.4.3: strict JSON, size limit,
JSON pointer resolution, exact quote matching, quote length and cross instance
duplication.  It cannot judge whether the evidence is convincing; it only proves
the citations can be located and matched.

    python3 tools/validate_material.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "agent" / "conversation.json"
MAX_BYTES = 128 * 1024
MAX_QUOTE = 240


def strict_load(text):
    def reject_constant(value):
        raise ValueError("non standard JSON constant: %s" % value)

    def reject_duplicates(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                raise ValueError("duplicate key: %s" % key)
            seen[key] = value
        return seen

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)


def resolve(document, pointer):
    if not pointer.startswith("/"):
        raise ValueError("pointer must start with '/': %r" % pointer)
    node = document
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise KeyError("pointer goes through a scalar at %r" % token)
    return node


def main():
    raw = TARGET.read_bytes()
    problems = []
    if len(raw) > MAX_BYTES:
        problems.append("file is %d bytes, limit is %d" % (len(raw), MAX_BYTES))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        print("FAIL: not valid UTF-8: %s" % exc)
        return 1
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM")
    try:
        document = strict_load(text)
    except ValueError as exc:
        print("FAIL: invalid JSON: %s" % exc)
        return 1
    if not isinstance(document, dict):
        problems.append("top level must be an object")

    seen_quotes = {}
    checked = 0
    for index, instance in enumerate(document.get("instances", [])):
        ident = instance.get("id", "#%d" % index)
        for field in ("student_evidence", "result_evidence"):
            for item in instance.get(field, []):
                pointer, quote = item.get("pointer"), item.get("quote")
                if not isinstance(pointer, str) or not isinstance(quote, str):
                    problems.append("%s/%s: pointer and quote must be strings" % (ident, field))
                    continue
                if len(quote) > MAX_QUOTE:
                    problems.append("%s: quote longer than %d characters" % (ident, MAX_QUOTE))
                try:
                    target = resolve(document, pointer)
                except (KeyError, IndexError, ValueError) as exc:
                    problems.append("%s: pointer %s does not resolve (%s)" % (ident, pointer, exc))
                    continue
                if not isinstance(target, str):
                    problems.append("%s: %s is not a string field" % (ident, pointer))
                    continue
                if quote not in target:
                    problems.append("%s: quote not found at %s" % (ident, pointer))
                    continue
                if quote in seen_quotes:
                    problems.append(
                        "%s: quote already used by %s" % (ident, seen_quotes[quote])
                    )
                    continue
                seen_quotes[quote] = ident
                checked += 1

    secrets = ("github_pat_", "ghp_", "Bearer ")
    for needle in secrets:
        if needle in text:
            problems.append("material still contains a credential fragment: %s" % needle)

    if problems:
        for problem in problems:
            print("FAIL: " + problem)
        return 1
    print(
        "OK: %d bytes, %d evidence citations resolved and matched, no duplicate quote"
        % (len(raw), checked)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
