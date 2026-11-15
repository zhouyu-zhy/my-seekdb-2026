#!/usr/bin/env python3
"""Time an offline vector-index build and its first query on a dedicated test table."""

import argparse
import json
import math
import re
import statistics
import subprocess
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--table", required=True, help="Preloaded table with id BIGINT PK and v VECTOR")
    parser.add_argument("--queries", type=Path, required=True, help="One JSON vector per line, or id<TAB>JSON lines")
    parser.add_argument("--library", choices=("vsag", "cuvs"), required=True)
    parser.add_argument("--label", required=True, help="Build label, for example v5-cuvs, candidate-cuvs, or candidate-vsag")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--user", default="root")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) for value in (args.database, args.table)):
        parser.error("database and table must be simple identifiers")
    if args.repeats < 1:
        parser.error("repeats must be positive")

    def query(statement):
        client = ["mysql", "--batch", "--skip-column-names", "--raw", "--silent",
                  "--host", args.host, "--port", str(args.port), "--user", args.user, args.database]
        completed = subprocess.run(
            client, input=f"SET ob_query_timeout=900000000;\n{statement}\n",
            capture_output=True, text=True, timeout=950, check=True,
        )
        return completed.stdout.splitlines()

    vectors = []
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        vector = json.loads(line.rsplit("\t", 1)[-1])
        if not isinstance(vector, list) or not vector or not all(
            type(x) in (int, float) and math.isfinite(x) for x in vector
        ):
            parser.error("each query must be a nonempty finite JSON vector")
        vectors.append(vector)
    if not vectors or any(len(v) != len(vectors[0]) for v in vectors):
        parser.error("query file must contain vectors of one dimension")

    table = f"`{args.table}`"
    if any(row.split("\t")[2] != "PRIMARY" for row in query(f"SHOW INDEX FROM {table};")):
        parser.error("start with a dedicated table that has only its primary index")
    count, = query(f"SELECT COUNT(*) FROM {table};")
    if int(count) < 100:
        parser.error("the table needs at least 100 vectors")
    literals = [json.dumps(v, separators=(",", ":"), allow_nan=False) for v in vectors]

    def nearest(vector, k, approximate):
        mode = "APPROXIMATE" if approximate else ""
        return f"SELECT id FROM {table} ORDER BY l2_distance(v, '{vector}') {mode} LIMIT {k};"

    ground_truth = [query(nearest(v, 100, False)) for v in literals]
    if any(len(ids) != 100 or len(set(ids)) != 100 for ids in ground_truth):
        raise RuntimeError("invalid exact top-100 results")
    samples = []
    for _ in range(args.repeats):
        index = f"bench_{uuid.uuid4().hex[:12]}"
        try:
            started = time.perf_counter()
            first = query(
                f"CREATE VECTOR INDEX {index} ON {table}(v) WITH "
                f"(distance=l2,type=hnsw,lib={args.library},m=16,ef_construction=200,ef_search=64);\n"
                + nearest(literals[0], 10, True)
            )
            elapsed = time.perf_counter() - started
            if len(first) != 10 or len(set(first)) != 10:
                raise RuntimeError("first query failed cardinality or uniqueness")
            sample = {"build_and_first_query_seconds": elapsed}
            for k in (10, 100):
                started = time.perf_counter()
                returned = query("\n".join(nearest(vector, k, True) for vector in literals))
                seconds = time.perf_counter() - started
                if len(returned) != len(vectors) * k:
                    raise RuntimeError(f"wrong ID count for top-{k}")
                recall = 0.0
                for pos, exact in enumerate(ground_truth):
                    ids = set(returned[pos * k:(pos + 1) * k])
                    if len(ids) != k:
                        raise RuntimeError(f"duplicate IDs for top-{k}, probe {pos}")
                    recall += len(ids.intersection(exact[:k])) / k
                sample[f"top{k}"] = {"recall": recall / len(vectors), "qps": len(vectors) / seconds}
            samples.append(sample)
        finally:
            if any(row.split("\t")[2] == index for row in query(f"SHOW INDEX FROM {table};")):
                query(f"DROP INDEX {index} ON {table};")
    print(json.dumps({
        "label": args.label, "library": args.library, "server": query("SELECT VERSION();")[0],
        "rows": int(count), "dimension": len(vectors[0]), "queries_per_top_k": len(vectors),
        "median_build_and_first_query_seconds": statistics.median(
            sample["build_and_first_query_seconds"] for sample in samples
        ),
        "samples": samples,
    }, indent=2))


if __name__ == "__main__":
    main()
