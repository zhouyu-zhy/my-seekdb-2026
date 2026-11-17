"""Single-binary benchmark for the MiniVectorDB performance workload.

Measures the stages the platform scores: a 100-row-per-batch load, exact TopK
queries (half of them filtered), disk bytes and peak RSS.  All input text is
generated before the clock starts.

This is a local diagnostic: the numbers are not contest E values.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time


def request(sql, number=1):
    return (json.dumps({"id": number, "op": "execute", "sql": sql},
                       separators=(",", ":")) + "\n").encode()


def exchange(proc, message):
    proc.stdin.write(message)
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("database exited early")
    response = json.loads(line)
    if not response.get("ok"):
        raise RuntimeError(f"query failed: {response}")
    return response.get("rows", [])


def usage(proc):
    io = dict(line.split(":") for line in Path(f"/proc/{proc.pid}/io").read_text().splitlines())
    status = Path(f"/proc/{proc.pid}/status").read_text().splitlines()
    stat = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"read_bytes": int(io["read_bytes"]), "write_bytes": int(io["write_bytes"]),
            "cpu_sec": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
            "peak_rss_kib": next(int(s.split()[1]) for s in status if s.startswith("VmHWM:"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=920026)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    def vector():
        return "'[" + ",".join(str(round(rng.uniform(-1, 1), 7)) for _ in range(args.dim)) + "]'"

    values = [f"({i},'g{i % args.groups}',{vector()})" for i in range(args.rows)]
    writes = [request("INSERT INTO d VALUES " + ",".join(values[j:j + args.batch]))
              for j in range(0, args.rows, args.batch)]
    queries = [request(f"SELECT id,L2_DISTANCE(v,{vector()}) AS dist FROM d" +
                       (f" WHERE g='g{j % args.groups}'" if j % 2 else "") +
                       " ORDER BY dist LIMIT 10")
               for j in range(args.queries)]

    with tempfile.TemporaryDirectory() as directory:
        proc = subprocess.Popen([str(Path(args.binary).resolve()), "--data-dir", directory],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            ready = json.loads(proc.stdout.readline())
            assert ready == {"type": "ready"}, ready
            exchange(proc, request(f"CREATE TABLE d(id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR({args.dim}))"))
            before = usage(proc)
            start = time.perf_counter()
            for message in writes:
                exchange(proc, message)
            write_sec = time.perf_counter() - start
            after_write = usage(proc)
            exchange(proc, request("CREATE VECTOR INDEX ix ON d(v)"))
            cold_start = time.perf_counter()
            for message in queries[:3]:
                exchange(proc, message)
            cold_sec = time.perf_counter() - cold_start
            before_q = usage(proc)
            start = time.perf_counter()
            rows = 0
            for message in queries:
                rows += len(exchange(proc, message))
            query_sec = time.perf_counter() - start
            after_q = usage(proc)
            exchange(proc, b'{"id":0,"op":"shutdown"}\n')
            assert proc.wait(timeout=20) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=20)

    report = {
        "binary": str(args.binary),
        "parameters": {k: getattr(args, k) for k in ("rows", "dim", "queries", "batch", "seed", "groups")},
        "write": {"wall_sec": write_sec,
                  "write_bytes": after_write["write_bytes"] - before["write_bytes"],
                  "read_bytes": after_write["read_bytes"] - before["read_bytes"],
                  "cpu_sec": after_write["cpu_sec"] - before["cpu_sec"],
                  "peak_rss_kib": after_write["peak_rss_kib"]},
        "query": {"wall_sec": query_sec, "rows_returned": rows, "cold_3_sec": cold_sec,
                  "read_bytes": after_q["read_bytes"] - before_q["read_bytes"],
                  "write_bytes": after_q["write_bytes"] - before_q["write_bytes"],
                  "cpu_sec": after_q["cpu_sec"] - before_q["cpu_sec"],
                  "peak_rss_kib": after_q["peak_rss_kib"]},
        "startup_rss_kib": before["peak_rss_kib"],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()
