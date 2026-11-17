#!/usr/bin/env python3
"""Force-kill recovery checks that the public package does not cover.

The public mysqltest package has no directive for SIGKILL, so this script drives
the same JSONL protocol itself and kills the database at the points the task
description describes as the crash model:

1. a committed autocommit row must survive,
2. an open transaction must be gone after the kill,
3. a transaction killed around COMMIT is either fully there or fully absent,
4. killing the process again while it is recovering must not corrupt data.

Every scenario starts from a fresh data directory.  Run it on Linux::

    python3 tools/crash_check.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protocol_check"))

from _transport import Session  # noqa: E402

CONFIG = json.loads((ROOT / "protocol_check" / "config.json").read_text())


def command(python_exe):
    return [python_exe, str(ROOT / "src" / "minivectordb" / "__main__.py"), "--data-dir", "{data_dir}"]


def rows(session, sql, mode="exact"):
    response = session.request(sql, mode)
    if not response.get("ok"):
        raise AssertionError("statement failed: %r" % (response,))
    return response.get("rows", [])


def scenario(name, body):
    work = ROOT / "results" / "crash"
    work.mkdir(parents=True, exist_ok=True)
    log = work / (name + ".stderr.log")
    session = Session(CONFIG, str(log), command=ARGS.command)
    try:
        detail = body(session)
    finally:
        session.close()
    print("%-26s PASS  %s" % (name, detail))


def setup_table(session):
    session.start()
    session.request("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                    "content VARCHAR(32), embedding VECTOR(4));")


def case_committed_survives(session):
    setup_table(session)
    session.request("INSERT INTO docs VALUES (1, 'a', 'committed', '[1,0,0,0]');")
    session.crash()
    session.start()
    got = rows(session, "SELECT id, content FROM docs ORDER BY id;")
    assert got == [[1, "committed"]], got
    return "row 1 survived SIGKILL, %d row(s) visible" % len(got)


def case_uncommitted_dropped(session):
    setup_table(session)
    session.request("BEGIN;")
    session.request("INSERT INTO docs VALUES (1, 'a', 'pending', '[1,0,0,0]');")
    session.request("INSERT INTO docs VALUES (2, 'a', 'pending', '[0,1,0,0]');")
    assert len(rows(session, "SELECT id FROM docs ORDER BY id;")) == 2
    session.crash()
    session.start()
    got = rows(session, "SELECT id FROM docs ORDER BY id;")
    assert got == [], got
    return "open transaction dropped, 0 rows visible"


def case_commit_confirmed(session):
    setup_table(session)
    session.request("BEGIN;")
    session.request("INSERT INTO docs VALUES " + ",".join(
        "(%d, 'r', 'row', '[%d,0,0,0]')" % (i, i) for i in range(1, 101)) + ";")
    assert session.request("COMMIT;")["ok"] is True
    session.crash()
    session.start()
    got = rows(session, "SELECT id FROM docs ORDER BY id;")
    assert len(got) == 100, len(got)
    return "confirmed commit kept all 100 rows"


def case_commit_race(session):
    setup_table(session)
    session.request("BEGIN;")
    session.request("INSERT INTO docs VALUES " + ",".join(
        "(%d, 'r', 'row', '[%d,0,0,0]')" % (i, i) for i in range(1, 101)) + ";")
    # Send COMMIT and kill immediately: the reply may never arrive, and either
    # "fully committed" or "fully absent" is acceptable, never a partial set.
    session.request("COMMIT;", wait=False)
    session.crash()
    session.start()
    got = rows(session, "SELECT id FROM docs ORDER BY id;")
    count = len(got)
    assert count in (0, 100), "partial commit: %d rows" % count
    return "%d rows visible, expected 0 or 100" % count


def case_kill_during_recovery(session):
    setup_table(session)
    session.request("INSERT INTO docs VALUES (7, 'a', 'kept', '[0,0,0,1]');")
    session.crash()
    session.start(ready=False)
    time.sleep(0.05)
    session.crash()
    session.start()
    got = rows(session, "SELECT id, content FROM docs ORDER BY id;")
    assert got == [[7, "kept"]], got
    session.request("INSERT INTO docs VALUES (8, 'a', 'after', '[0,0,0,1]');")
    session.crash()
    session.start()
    got = rows(session, "SELECT id FROM docs ORDER BY id;")
    assert got == [[7], [8]], got
    return "recovery restart is repeatable, 2 rows"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", default=sys.executable)
    global ARGS
    ARGS = parser.parse_args()
    ARGS.command = command(ARGS.python_exe)
    for name, body in (
        ("committed_survives", case_committed_survives),
        ("uncommitted_dropped", case_uncommitted_dropped),
        ("commit_confirmed", case_commit_confirmed),
        ("kill_during_recovery", case_kill_during_recovery),
    ):
        scenario(name, body)
    for attempt in range(1, 6):
        scenario("commit_race_%d" % attempt, case_commit_race)
    return 0


if __name__ == "__main__":
    sys.exit(main())
