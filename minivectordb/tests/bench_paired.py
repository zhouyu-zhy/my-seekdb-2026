"""Reproducible paired benchmark; all input generation is outside timing.

The values are local measurements, not contest E or rank predictions.
Run baseline/candidate alternately on the same machine without other builds.
"""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time


def request(sql, number=1):
    return (json.dumps({"id": number, "op": "execute", "sql": sql},
                       separators=(",", ":")) + "\n").encode()


def exchange(proc, message):
    proc.stdin.write(message)
    proc.stdin.flush()
    response = json.loads(proc.stdout.readline())
    assert response["ok"], response
    return response.get("rows", [])


def usage(proc):
    io = dict(line.split(":") for line in Path(f"/proc/{proc.pid}/io").read_text().splitlines())
    status = Path(f"/proc/{proc.pid}/status").read_text().splitlines()
    # Linux stat: after the parenthesized comm, indexes 11/12 are utime/stime.
    stat = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"read_bytes": int(io["read_bytes"]), "write_bytes": int(io["write_bytes"]),
            "cpu_sec": (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK"),
            "peak_rss_kib": next(int(s.split()[1]) for s in status if s.startswith("VmHWM:"))}


def stage(proc, inputs):
    start_usage = usage(proc)
    start = time.perf_counter()
    rows = [exchange(proc, message) for message in inputs]
    seconds = time.perf_counter() - start
    end_usage = usage(proc)
    return rows, {"wall_sec": seconds,
                  **{key: end_usage[key] - start_usage[key]
                     for key in ("read_bytes", "write_bytes", "cpu_sec")},
                  "peak_rss_kib": end_usage["peak_rss_kib"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=920026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.rounds >= 3 and args.rows > 0 and args.dim > 0 and args.queries > 0
    rng = random.Random(args.seed)

    def vector():
        return "'[" + ",".join(str(round(rng.uniform(-1, 1), 7)) for _ in range(args.dim)) + "]'"

    values = [f"({i},'g{i%7}',{vector()})" for i in range(args.rows)]
    writes = [request("INSERT INTO d VALUES " + ",".join(values[j:j+100]))
              for j in range(0, args.rows, 100)]
    queries = [request(f"SELECT id,L2_DISTANCE(v,{vector()}) AS dist FROM d" +
                       (f" WHERE g='g{j%7}'" if j % 2 else "") + " ORDER BY dist LIMIT 10")
               for j in range(args.queries)]
    samples = []
    reference = None
    for iteration in range(args.rounds):
        order = ["baseline", "candidate"] if iteration % 2 == 0 else ["candidate", "baseline"]
        for label in order:
            binary = str(Path(getattr(args, label)).resolve())
            with tempfile.TemporaryDirectory() as directory:
                proc = subprocess.Popen([binary, "--data-dir", directory],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                try:
                    assert json.loads(proc.stdout.readline())["type"] == "ready"
                    exchange(proc, request(f"CREATE TABLE d(id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR({args.dim}))"))
                    _, writing = stage(proc, writes)
                    exchange(proc, request("CREATE VECTOR INDEX ix ON d(v)"))
                    _, cold = stage(proc, queries[:1])
                    results, reading = stage(proc, queries)
                    if reference is None:
                        reference = results
                    else:
                        assert results == reference, (iteration, label, "exact result mismatch")
                    sample = {"round": iteration + 1, "label": label, "write": writing,
                              "cold_query": cold, "query": reading}
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
                    exchange(proc, b'{"id":0,"op":"shutdown"}\n')
                    assert proc.wait(timeout=15) == 0
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=15)
    medians = {label: {stage_name: {key: statistics.median(
        sample[stage_name][key] for sample in samples if sample["label"] == label)
        for key in samples[0][stage_name]} for stage_name in ("write", "cold_query", "query")}
        for label in ("baseline", "candidate")}
    report = {"parameters": {**vars(args), "output": str(args.output)},
              "samples": samples, "medians": medians,
              "all_exact_responses_match": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"medians": medians}, indent=2))


if __name__ == "__main__":
    main()
