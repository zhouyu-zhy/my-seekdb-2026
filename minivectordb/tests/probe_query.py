"""Timing probe: request overhead versus the vector TopK scan path."""
import json
import random
import subprocess
import sys
import tempfile
import time

BIN = sys.argv[1] if len(sys.argv) > 1 else "./minivectordb-avx2"
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
DIM = 128
QUERIES = 1000


def req(sql, n=1):
    return (json.dumps({"id": n, "op": "execute", "sql": sql}, separators=(",", ":")) + "\n").encode()


def exchange(p, msg):
    p.stdin.write(msg)
    p.stdin.flush()
    return json.loads(p.stdout.readline())


rng = random.Random(7)
vec = lambda: "'[" + ",".join(str(round(rng.uniform(-1, 1), 7)) for _ in range(DIM)) + "]'"
values = [f"({i},'g{i%7}',{vec()})" for i in range(ROWS)]
writes = [req("INSERT INTO d VALUES " + ",".join(values[j:j + 100])) for j in range(0, ROWS, 100)]
where = lambda j: f" WHERE g='g{j % 7}'" if j % 2 else ""
topk = [req(f"SELECT id,L2_DISTANCE(v,{vec()}) AS dist FROM d{where(j)} ORDER BY dist LIMIT 10")
        for j in range(QUERIES)]
# A 1e-30 component disables the float shortlist and forces the generic executor.
tiny_vec = "'[1e-30," + ",".join(["0"] * (DIM - 1)) + "]'"
slow = [req(f"SELECT id,L2_DISTANCE(v,{tiny_vec}) AS dist FROM d{where(j)} ORDER BY dist LIMIT 10")
        for j in range(QUERIES)]
noop = [req("SELECT id FROM d LIMIT 0") for _ in range(QUERIES)]
point = [req("SELECT id FROM d WHERE id = 999999") for _ in range(QUERIES)]

with tempfile.TemporaryDirectory() as data:
    p = subprocess.Popen([BIN, "--data-dir", data], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    json.loads(p.stdout.readline())
    exchange(p, req("CREATE TABLE d(id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR(128))"))
    for m in writes:
        exchange(p, m)
    exchange(p, req("CREATE VECTOR INDEX ix ON d(v)"))

    def run(label, messages):
        start = time.perf_counter()
        for m in messages:
            r = exchange(p, m)
            assert r["ok"], r
        sec = time.perf_counter() - start
        print(f"{label:12s} {sec * 1000:8.2f} ms total  {sec * 1e6 / len(messages):7.2f} us/request", flush=True)

    run("noop", noop)
    run("point", point)
    run("topk-fast", topk)
    run("topk-generic", slow)
    run("topk-fast2", topk)
    p.stdin.write(b'{"id":0,"op":"shutdown"}\n')
    p.stdin.flush()
    p.wait(timeout=10)
