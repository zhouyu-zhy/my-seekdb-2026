#!/usr/bin/env python3
"""Checks for task requirements that the 20 public cases do not exercise.

Each check names the requirement from the task description (rejection of invalid
vectors, table level primary key, ``SELECT *``, statement atomicity, restart with
a torn file tail, and so on).  Run on Linux::

    python3 tools/spec_check.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protocol_check"))

from _transport import Session  # noqa: E402

CONFIG = json.loads((ROOT / "protocol_check" / "config.json").read_text())
CHECKS = []


def check(name):
    def decorate(function):
        CHECKS.append((name, function))
        return function

    return decorate


def run(session, sql, mode="exact"):
    return session.request(sql, mode)


def ok(session, sql, mode="exact"):
    response = run(session, sql, mode)
    assert response.get("ok") is True, "%s -> %r" % (sql, response)
    return response


def fails(session, sql, mode="exact"):
    response = run(session, sql, mode)
    assert response.get("ok") is False, "%s should fail, got %r" % (sql, response)
    assert isinstance(response["error"]["code"], str) and response["error"]["code"]
    return response["error"]["code"]


@check("table_level_primary_key_and_select_star")
def _(session):
    ok(session, "CREATE TABLE t (id BIGINT, c VARCHAR(8), v VECTOR(2), PRIMARY KEY (id));")
    ok(session, "INSERT INTO t VALUES (2,'b','[0,1]'),(1,'a','[1,0]');")
    response = ok(session, "SELECT * FROM t ORDER BY id;")
    assert response["columns"] == ["id", "c", "v"], response["columns"]
    assert response["rows"] == [[1, "a", [1.0, 0.0]], [2, "b", [0.0, 1.0]]], response["rows"]


@check("reject_invalid_vectors")
def _(session):
    ok(session, "CREATE TABLE v (id BIGINT PRIMARY KEY, e VECTOR(2));")
    assert fails(session, "INSERT INTO v VALUES (1,'[nan,0]');") == "invalid_vector"
    assert fails(session, "INSERT INTO v VALUES (1,'[inf,0]');") == "invalid_vector"
    assert fails(session, "INSERT INTO v VALUES (1,'[1e39,0]');") == "invalid_vector"
    assert fails(session, "INSERT INTO v VALUES (1,'[1]');") == "dimension_mismatch"
    assert fails(session, "INSERT INTO v VALUES (1,'[a,0]');") == "invalid_vector"
    assert fails(session, "SELECT id FROM v WHERE id = 1 AND e = '[1,0]';") == "unsupported_sql"
    rows = ok(session, "SELECT id FROM v;")["rows"]
    assert rows == [], rows


@check("query_vector_must_match_dimension")
def _(session):
    ok(session, "CREATE TABLE v (id BIGINT PRIMARY KEY, e VECTOR(3));")
    ok(session, "INSERT INTO v VALUES (1,'[1,2,3]');")
    assert fails(session, "SELECT L2_DISTANCE(e,'[1,2]') AS d FROM v ORDER BY d LIMIT 1;") == \
        "dimension_mismatch"


@check("varchar_length_counts_unicode_characters")
def _(session):
    ok(session, "CREATE TABLE s (id BIGINT PRIMARY KEY, c VARCHAR(2));")
    ok(session, "INSERT INTO s VALUES (1,'中文');")
    assert fails(session, "INSERT INTO s VALUES (2,'三个字');") == "value_too_long"
    rows = ok(session, "SELECT c FROM s;")["rows"]
    assert rows == [["中文"]], rows


@check("string_order_is_code_point_order")
def _(session):
    ok(session, "CREATE TABLE s (id BIGINT PRIMARY KEY, c VARCHAR(8));")
    ok(session, "INSERT INTO s VALUES (1,'a'),(2,'中'),(3,'z');")
    rows = ok(session, "SELECT c FROM s WHERE c > 'a' ORDER BY c;")["rows"]
    assert rows == [["z"], ["中"]], rows


@check("statement_atomicity_single_insert")
def _(session):
    ok(session, "CREATE TABLE a (id BIGINT PRIMARY KEY, c VARCHAR(4));")
    ok(session, "INSERT INTO a VALUES (1,'x');")
    assert fails(session, "INSERT INTO a VALUES (2,'y'),(1,'dup');") == "duplicate_key"
    rows = ok(session, "SELECT id FROM a ORDER BY id;")["rows"]
    assert rows == [[1]], rows


@check("autocommit_failure_leaves_nothing")
def _(session):
    ok(session, "CREATE TABLE a (id BIGINT PRIMARY KEY, e VECTOR(2));")
    assert fails(session, "INSERT INTO a VALUES (1,'[0,0]'),(2,'[0]');") == "dimension_mismatch"
    assert ok(session, "SELECT id FROM a;")["rows"] == []


@check("unsupported_statements_are_rejected")
def _(session):
    ok(session, "CREATE TABLE a (id BIGINT PRIMARY KEY, c VARCHAR(4));")
    # The task only requires a defined error, not a specific code for these.
    assert fails(session, "ALTER TABLE a ADD extra BIGINT;") == "unsupported_sql"
    assert fails(session, "DROP TABLE a;") == "unsupported_sql"
    assert fails(session, "SELECT id FROM a JOIN a AS b ON a.id = b.id;")
    assert fails(session, "SELECT COUNT(*) FROM a;") == "unsupported_sql"
    assert fails(session, "SELECT id FROM a WHERE id = 1 OR id = 2;") == "unsupported_sql"
    assert fails(session, "SELECT id FROM a UNION SELECT id FROM a;")
    assert fails(session, "SELECT missing FROM a;") == "unknown_column"
    ok(session, "BEGIN;")
    assert fails(session, "BEGIN;") == "unsupported_sql"
    ok(session, "ROLLBACK;")


@check("multiple_statements_in_one_request")
def _(session):
    ok(session, "CREATE TABLE a (id BIGINT PRIMARY KEY);")
    fails(session, "INSERT INTO a VALUES (1); INSERT INTO a VALUES (2);")
    rows = ok(session, "SELECT id FROM a;")["rows"]
    assert rows == [], rows


@check("session_survives_errors")
def _(session):
    ok(session, "CREATE TABLE a (id BIGINT PRIMARY KEY);")
    fails(session, "SELECT missing FROM a;")
    fails(session, "SELECT id FROM missing_table;")
    fails(session, "this is not sql;")
    ok(session, "INSERT INTO a VALUES (1);")
    assert ok(session, "SELECT id FROM a;")["rows"] == [[1]]


@check("update_and_delete_without_matches")
def _(session):
    ok(session, "CREATE TABLE a (id BIGINT PRIMARY KEY, c VARCHAR(4));")
    ok(session, "INSERT INTO a VALUES (1,'x'),(2,'y');")
    assert ok(session, "UPDATE a SET c = 'z' WHERE id = 99;")["affected_rows"] == 0
    assert ok(session, "DELETE FROM a WHERE id > 100;")["affected_rows"] == 0
    assert ok(session, "DELETE FROM a;")["affected_rows"] == 2
    assert ok(session, "SELECT id FROM a;")["rows"] == []


@check("order_desc_limit_zero_and_ties")
def _(session):
    ok(session, "CREATE TABLE o (id BIGINT PRIMARY KEY, c VARCHAR(4), n BIGINT);")
    ok(session, "INSERT INTO o VALUES (1,'b',5),(2,'a',5),(3,'c',7);")
    rows = ok(session, "SELECT id FROM o ORDER BY n;")["rows"]
    assert rows == [[1], [2], [3]], rows  # tie on n=5 resolved by primary key
    rows = ok(session, "SELECT id FROM o ORDER BY n DESC LIMIT 2;")["rows"]
    assert rows == [[3], [1]], rows
    rows = ok(session, "SELECT id FROM o ORDER BY c DESC;")["rows"]
    assert rows == [[3], [1], [2]], rows
    response = ok(session, "SELECT id FROM o ORDER BY id LIMIT 0;")
    assert response["rows"] == [] and response["columns"] == ["id"], response


@check("filtered_topk_never_prefilters_globally")
def _(session):
    ok(session, "CREATE TABLE f (id BIGINT PRIMARY KEY, g VARCHAR(8), e VECTOR(2));")
    ok(session, "INSERT INTO f VALUES (1,'keep','[5,0]'),(2,'other','[0,0]'),"
                "(3,'other','[0.1,0]'),(4,'other','[0.2,0]'),(5,'keep','[9,0]');")
    rows = ok(session, "SELECT id FROM f WHERE g = 'keep' ORDER BY "
                       "L2_DISTANCE(e,'[5,0]') LIMIT 1;")["rows"]
    assert rows == [[1]], rows


@check("transaction_sees_own_vector_writes")
def _(session):
    ok(session, "CREATE TABLE tx (id BIGINT PRIMARY KEY, e VECTOR(2));")
    ok(session, "INSERT INTO tx VALUES (1,'[1,0]');")
    ok(session, "BEGIN;")
    ok(session, "UPDATE tx SET e = '[0,1]' WHERE id = 1;")
    rows = ok(session, "SELECT L2_DISTANCE(e,'[0,1]') AS d FROM tx ORDER BY d;")["rows"]
    assert abs(rows[0][0]) < 1e-6, rows
    ok(session, "ROLLBACK;")
    rows = ok(session, "SELECT L2_DISTANCE(e,'[1,0]') AS d FROM tx ORDER BY d;")["rows"]
    assert abs(rows[0][0]) < 1e-6, rows


@check("torn_tail_is_truncated_on_restart")
def _(session):
    ok(session, "CREATE TABLE tt (id BIGINT PRIMARY KEY, c VARCHAR(4));")
    ok(session, "INSERT INTO tt VALUES (1,'x');")
    session.shutdown()
    data_dir = session.temp.name
    victim = None
    for name in os.listdir(data_dir):
        if name.startswith("table_") and name.endswith(".dat"):
            victim = os.path.join(data_dir, name)
    assert victim, os.listdir(data_dir)
    with open(victim, "ab") as handle:
        handle.write(b"\x00\x01\x02\x03\x04")  # half written frame
    session.start()
    rows = ok(session, "SELECT id, c FROM tt;")["rows"]
    assert rows == [[1, "x"]], rows
    ok(session, "INSERT INTO tt VALUES (2,'y');")
    rows = ok(session, "SELECT id FROM tt ORDER BY id;")["rows"]
    assert rows == [[1], [2]], rows


def main():
    command = [sys.executable, str(ROOT / "src" / "minivectordb" / "__main__.py"),
               "--data-dir", "{data_dir}"]
    failures = 0
    log_dir = ROOT / "results" / "spec"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, body in CHECKS:
        session = Session(CONFIG, str(log_dir / (name + ".log")), command=command)
        try:
            session.start()
            body(session)
            print("%-46s PASS" % name)
        except Exception as exc:
            failures += 1
            print("%-46s FAIL  %s" % (name, exc))
        finally:
            session.close()
    print("%d/%d checks passed" % (len(CHECKS) - failures, len(CHECKS)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
