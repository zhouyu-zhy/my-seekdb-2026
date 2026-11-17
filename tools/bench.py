#!/usr/bin/env python3
"""Small end-to-end benchmark shaped like the official performance workload.

* 10,000 rows, 128 dimension float32 vectors, 10 categories,
* load in 100 autocommit batches of 100 rows,
* 1,000 TopK queries with K=10, half of them filtered by ``category``,
* the same 1,000 queries again in ``ann`` mode to measure Recall@10.

Wall clock is measured around the protocol round trips (load and query phase
separately), peak RSS is read from ``/proc/<pid>/status`` of the server process.
Run on Linux::

    python3 tools/bench.py --label columnar
    MVDB_DISABLE_COLUMNAR=1 python3 tools/bench.py --label rows
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "protocol_check"))

from _transport import Session  # noqa: E402

CONFIG = json.loads((ROOT / "protocol_check" / "config.json").read_text())
ROWS = 10_000
DIM = 128
K = 10
QUERIES = 1_000
CATEGORIES = ["c%02d" % i for i in range(10)]


def peak_rss_kb(pid):
    try:
        with open("/proc/%d/status" % pid, "r") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def vector_literal(rng):
    return "[" + ",".join("%.6f" % rng.uniform(-1.0, 1.0) for _ in range(DIM)) + "]"


def directory_bytes(path):
    total = 0
    for entry in Path(path).rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run")
    parser.add_argument("--python-exe", default=sys.executable)
    arguments = parser.parse_args()
    rng = random.Random(20261117)
    log = ROOT / "results" / "bench" / (arguments.label + ".stderr.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    session = Session(
        CONFIG,
        str(log),
        command=[arguments.python_exe, str(ROOT / "src" / "minivectordb" / "__main__.py"),
                 "--data-dir", "{data_dir}"],
    )
    try:
        session.start()
        session.request("CREATE TABLE bench (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                        "embedding VECTOR(%d));" % DIM)
        batch = []
        started = time.perf_counter()
        for index in range(1, ROWS + 1):
            batch.append("(%d, '%s', '%s')" % (index, CATEGORIES[index % 10], vector_literal(rng)))
            if len(batch) == 100:
                session.request("INSERT INTO bench VALUES " + ",".join(batch) + ";")
                batch = []
        load_seconds = time.perf_counter() - started
        load_rss = peak_rss_kb(session.proc.pid)

        queries = []
        for index in range(QUERIES):
            target = vector_literal(rng)
            if index % 2:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM bench "
                       "WHERE category = '%s' ORDER BY distance LIMIT %d;" %
                       (target, CATEGORIES[index % 10], K))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM bench "
                       "ORDER BY distance LIMIT %d;" % (target, K))
            queries.append(sql)

        started = time.perf_counter()
        exact = [session.request(sql, "exact")["rows"] for sql in queries]
        query_seconds = time.perf_counter() - started
        query_rss = peak_rss_kb(session.proc.pid)

        started = time.perf_counter()
        approximate = [session.request(sql, "ann")["rows"] for sql in queries]
        ann_seconds = time.perf_counter() - started

        overlap = 0
        total = 0
        for left, right in zip(exact, approximate):
            wanted = [row[0] for row in left]
            got = [row[0] for row in right]
            overlap += len(set(wanted) & set(got))
            total += len(wanted)
        recall = overlap / total if total else 1.0

        session.shutdown()
        summary = {
            "label": arguments.label,
            "rows": ROWS,
            "dim": DIM,
            "queries": QUERIES,
            "load_seconds": round(load_seconds, 3),
            "query_seconds": round(query_seconds, 3),
            "query_ms_per_query": round(query_seconds * 1000 / QUERIES, 2),
            "ann_seconds": round(ann_seconds, 3),
            "recall_at_10": round(recall, 4),
            "peak_rss_mb_load": None if load_rss is None else round(load_rss / 1024, 1),
            "peak_rss_mb_query": None if query_rss is None else round(query_rss / 1024, 1),
            "data_dir_mb": round(directory_bytes(session.temp.name) / 1048576, 1),
        }
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
