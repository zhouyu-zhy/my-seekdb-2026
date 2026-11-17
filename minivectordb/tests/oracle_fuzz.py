"""Randomised differential test: our engine vs an independent Python oracle.

Covers the documented SQL subset (BIGINT primary key, VARCHAR, VECTOR, CRUD,
WHERE/AND, ORDER BY, LIMIT, L2_DISTANCE TopK with and without a filter,
transactions) and compares every response.  The oracle is written from the
spec text, not from the implementation, so it catches common-mode bugs that a
comparison against our own older binary cannot.
"""
import json
import math
import random
import subprocess
import sys
import tempfile


class Oracle:
    def __init__(self):
        self.tables = {}      # name -> {"cols": [(name, type, len)], "pk": idx, "vec": idx}
        self.rows = {}        # name -> {pk: [values]}

    def create(self, name, cols, pk, vec):
        self.tables[name] = {"cols": cols, "pk": pk, "vec": vec}
        self.rows[name] = {}

    def insert(self, name, values):
        t = self.tables[name]
        for row in values:
            self.rows[name][row[t["pk"]]] = list(row)

    def delete(self, name, pk):
        self.rows[name].pop(pk, None)

    def update(self, name, pk, assign):
        t = self.tables[name]
        row = self.rows[name].get(pk)
        if row is None:
            return
        for idx, val in assign:
            row[idx] = val

    def where(self, name, preds):
        t = self.tables[name]
        out = []
        for pk in sorted(self.rows[name]):
            row = self.rows[name][pk]
            ok = True
            for (col, op, lit) in preds:
                v = row[col]
                if op == "=" and not (v == lit):
                    ok = False
                elif op == "<" and not (v < lit):
                    ok = False
                elif op == "<=" and not (v <= lit):
                    ok = False
                elif op == ">" and not (v > lit):
                    ok = False
                elif op == ">=" and not (v >= lit):
                    ok = False
                if not ok:
                    break
            if ok:
                out.append(row)
        return out

    def select(self, name, preds, order_col, desc, limit):
        rows = self.where(name, preds)
        t = self.tables[name]
        if order_col is None:
            rows = sorted(rows, key=lambda r: r[t["pk"]])
        else:
            rows = sorted(rows, key=lambda r: (r[order_col], r[t["pk"]]), reverse=desc)
            if desc:
                # ties still ascend by primary key
                rows.sort(key=lambda r: r[t["pk"]])
                rows.sort(key=lambda r: r[order_col], reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows

    def topk(self, name, query, preds, k):
        rows = self.where(name, preds)
        t = self.tables[name]
        scored = []
        for r in rows:
            v = r[t["vec"]]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, query)))
            scored.append((d, r[t["pk"]], r))
        scored.sort(key=lambda x: (x[0], x[1]))
        return scored[:k]


def req(sql, ident=1):
    return (json.dumps({"id": ident, "op": "execute", "sql": sql, "search_mode": "exact"},
                       separators=(",", ":")) + "\n").encode()


def fmt_vec(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-5 * max(1.0, abs(float(a)), abs(float(b)))
        except Exception:
            return False
    return a == b


def main():
    binary = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = random.Random(seed)
    dim = rng.choice([2, 4, 8])
    data = tempfile.mkdtemp(prefix="oracle-")

    proc = subprocess.Popen([binary, "--data-dir", data], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    assert json.loads(proc.stdout.readline()) == {"type": "ready"}

    def send(sql):
        proc.stdin.write(req(sql))
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    oracle = Oracle()
    cols = [("id", "BIGINT", 0), ("tag", "VARCHAR", 8), ("v", "VECTOR", dim)]
    r = send(f"CREATE TABLE t (id BIGINT PRIMARY KEY, tag VARCHAR(8), v VECTOR({dim}));")
    assert r.get("ok"), r
    oracle.create("t", cols, 0, 2)

    failures = 0
    next_pk = 1
    for step in range(steps):
        action = rng.choices(["insert", "select", "topk", "update", "delete", "select", "txn"],
                             weights=[28, 18, 22, 9, 9, 5, 9])[0]
        if action == "txn":
            committed = rng.random() < 0.5
            r = send("BEGIN;")
            if not r.get("ok"):
                print("BEGIN failed:", r)
                failures += 1
                continue
            pending = []
            for _ in range(rng.randint(1, 3)):
                kind = rng.choice(["insert", "update", "delete"])
                if kind == "insert":
                    pk = next_pk
                    next_pk += 1
                    tag = rng.choice(["a", "b", "x", "yy", ""])
                    vec = [round(rng.uniform(-3, 3), 3) for _ in range(dim)]
                    r = send(f"INSERT INTO t VALUES ({pk},'{tag}',{fmt_vec(vec)});")
                    if not r.get("ok"):
                        print("txn INSERT failed:", r)
                        failures += 1
                    pending.append(("insert", [pk, tag, vec]))
                elif kind == "update" and oracle.rows["t"]:
                    pk = rng.choice(sorted(oracle.rows["t"]))
                    tag = rng.choice(["x", "a", ""])
                    r = send(f"UPDATE t SET tag='{tag}' WHERE id={pk};")
                    if not r.get("ok"):
                        print("txn UPDATE failed:", r)
                        failures += 1
                    pending.append(("update", pk, tag))
                elif oracle.rows["t"]:
                    pk = rng.choice(sorted(oracle.rows["t"]))
                    r = send(f"DELETE FROM t WHERE id={pk};")
                    if not r.get("ok"):
                        print("txn DELETE failed:", r)
                        failures += 1
                    pending.append(("delete", pk))
            # inside the transaction our own writes must be visible
            r = send("SELECT id FROM t ORDER BY id;")
            visible = set(row[0] for row in r["rows"])
            expect = set(oracle.rows["t"])
            for item in pending:
                if item[0] == "insert":
                    expect.add(item[1][0])
                elif item[0] == "delete":
                    expect.discard(item[1])
            if visible != expect:
                print(f"TXN VISIBILITY mismatch: missing={sorted(expect - visible)[:5]} extra={sorted(visible - expect)[:5]}")
                failures += 1
            r = send("COMMIT;" if committed else "ROLLBACK;")
            if not r.get("ok"):
                print("txn end failed:", r)
                failures += 1
                continue
            if committed:
                for item in pending:
                    if item[0] == "insert":
                        oracle.insert("t", [item[1]])
                    elif item[0] == "update":
                        oracle.update("t", item[1], [(1, item[2])])
                    else:
                        oracle.delete("t", item[1])
            continue
        if action == "insert":
            batch = []
            for _ in range(rng.randint(1, 4)):
                pk = next_pk
                next_pk += 1
                tag = rng.choice(["a", "b", "c", "dd", ""])
                vec = [round(rng.uniform(-3, 3), 3) for _ in range(dim)]
                batch.append((pk, tag, vec))
            sql = "INSERT INTO t VALUES " + ",".join(
                f"({pk},'{tag}',{fmt_vec(vec)})" for pk, tag, vec in batch)
            r = send(sql)
            if not r.get("ok"):
                print("INSERT failed:", r)
                failures += 1
                continue
            oracle.insert("t", [[pk, tag, vec] for pk, tag, vec in batch])
        elif action in ("select", "topk"):
            preds = []
            sql_pred = ""
            if rng.random() < 0.5:
                tag = rng.choice(["a", "b", "c", "dd", ""])
                op = rng.choice(["=", "<", "<=", ">", ">="])
                preds.append((1, op, tag))
                sql_pred = f" WHERE tag {op} '{tag}'"
                if rng.random() < 0.3:
                    op2 = rng.choice(["=", "<", ">="])
                    tag2 = rng.choice(["a", "b", "c", "dd"])
                    preds.append((1, op2, tag2))
                    sql_pred += f" AND tag {op2} '{tag2}'"
            if action == "select":
                order = rng.choice([None, 0, 1])
                desc = rng.random() < 0.3
                limit = rng.choice([None, 1, 3, 10])
                sql = "SELECT id" + (", tag" if rng.random() < 0.5 else "")
                sql += " FROM t" + sql_pred
                if order is not None:
                    sql += f" ORDER BY {cols[order][0]}" + (" DESC" if desc else "")
                if limit is not None:
                    sql += f" LIMIT {limit}"
                sql += ";"
                r = send(sql)
                if not r.get("ok"):
                    print("SELECT failed:", sql, r)
                    failures += 1
                    continue
                ncol = len(r.get("columns") or [])
                got = [[row[0], row[1]] if ncol > 1 else [row[0]] for row in r["rows"]]
                want_rows = oracle.select("t", preds, order, desc, limit)
                want = []
                for row in want_rows:
                    item = [row[0]]
                    if ncol > 1:
                        item.append(row[1])
                    want.append(item)
                if order is None:
                    if limit is None:
                        # no ordering and no truncation: the whole set must match
                        same = sorted(map(str, got)) == sorted(map(str, want))
                    else:
                        # which rows a limit keeps is unspecified; only the size
                        # and membership of the returned rows are constrained
                        filtered = oracle.where("t", preds)
                        universe = {str([r[0]] + ([r[1]] if ncol > 1 else []))
                                    for r in filtered}
                        same = (len(got) == min(len(filtered), limit)
                                and all(str(g) in universe for g in got)
                                and len({str(g) for g in got}) == len(got))
                else:
                    same = got == want
                if not same:
                    print(f"MISMATCH {sql}\n  got  {got[:8]}\n  want {want[:8]}")
                    failures += 1
            else:
                q = [round(rng.uniform(-3, 3), 3) for _ in range(dim)]
                k = rng.choice([1, 3, 5, 10])
                sql = (f"SELECT id, L2_DISTANCE(v, {fmt_vec(q)}) AS d FROM t{sql_pred} "
                       f"ORDER BY d LIMIT {k};")
                r = send(sql)
                if not r.get("ok"):
                    print("TOPK failed:", sql, r)
                    failures += 1
                    continue
                got = [(row[0], row[1]) for row in r["rows"]]
                want = [(pk, d) for d, pk, _ in oracle.topk("t", q, preds, k)]
                if [g[0] for g in got] != [w[0] for w in want] or not all(
                        close(g[1], w[1]) for g, w in zip(got, want)):
                    print(f"MISMATCH {sql}\n  got  {got}\n  want {want}")
                    failures += 1
        elif action == "update":
            pks = sorted(oracle.rows["t"])
            if not pks:
                continue
            pk = rng.choice(pks)
            tag = rng.choice(["x", "yy", ""])
            sql = f"UPDATE t SET tag='{tag}' WHERE id={pk};"
            r = send(sql)
            if not r.get("ok"):
                print("UPDATE failed:", r)
                failures += 1
                continue
            oracle.update("t", pk, [(1, tag)])
        else:
            pks = sorted(oracle.rows["t"])
            if not pks:
                continue
            pk = rng.choice(pks)
            sql = f"DELETE FROM t WHERE id={pk};"
            r = send(sql)
            if not r.get("ok"):
                print("DELETE failed:", r)
                failures += 1
                continue
            oracle.delete("t", pk)

    # restart: committed state must survive
    proc.stdin.write(b'{"id":0,"op":"shutdown"}\n')
    proc.stdin.close()
    proc.wait(timeout=60)
    proc = subprocess.Popen([binary, "--data-dir", data], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    ready = json.loads(proc.stdout.readline())
    if ready != {"type": "ready"}:
        print("restart failed:", ready)
        failures += 1
    else:
        r = send("SELECT id, tag FROM t ORDER BY id;")
        got = [[row[0], row[1]] for row in r["rows"]]
        want = [[pk, oracle.rows["t"][pk][1]] for pk in sorted(oracle.rows["t"])]
        if got != want:
            print(f"RESTART MISMATCH\n  got  {got[:6]}\n  want {want[:6]}")
            failures += 1
    proc.stdin.write(b'{"id":0,"op":"shutdown"}\n')
    proc.stdin.close()
    proc.wait(timeout=60)
    print(f"seed={seed} steps={steps} rows={len(oracle.rows['t'])} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
