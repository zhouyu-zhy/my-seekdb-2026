"""Differential and independent checks for exact SIMD pruning and string scans."""
import argparse
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile

def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]

def literal(vector):
    return "'[" + ",".join(map(str, vector)) + "]'"

class Session:
    def __init__(self, binary, directory):
        self.proc = subprocess.Popen([str(Path(binary).resolve()), "--data-dir", directory],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert json.loads(self.proc.stdout.readline())["type"] == "ready"

    def raw(self, data):
        self.proc.stdin.write(data + b"\n")
        self.proc.stdin.flush()
        result = self.proc.stdout.readline()
        assert result, "database exited"
        return json.loads(result)

    def sql(self, sql):
        reply = self.raw(json.dumps({"id": 1, "sql": sql}, ensure_ascii=False).encode())
        assert reply["ok"], (sql[:100], reply)
        return reply.get("rows", [])

    def close(self):
        assert self.raw(b'{"id":0,"op":"shutdown"}')["ok"]
        assert self.proc.wait(timeout=15) == 0


def strings(binary):
    with tempfile.TemporaryDirectory() as directory:
        db = Session(binary, directory)
        db.sql("CREATE TABLE s(id BIGINT PRIMARY KEY, text VARCHAR(1000))")
        count = 0
        # Quotes, escapes and multibyte characters on both sides of every SIMD
        # block boundary; keep SQL and JSON escaping distinct.
        for prefix in range(97):
            for ending in ("'", "\\", "\n", "\t", "\x00", '中😀"'):
                text = "x" * prefix + ending + "tail"
                sql_text = text.replace("\\", "\\\\").replace("'", "''")
                db.sql(f"INSERT INTO s VALUES({count},'{sql_text}')")
                assert db.sql(f"SELECT text FROM s WHERE id={count}") == [[text]]
                count += 1
        for prefix in range(65):
            for byte in range(32):
                if byte in (10, 13):
                    continue  # line framing is independently tested by the public driver
                data = b'{"id":1,"sql":"' + b"x" * prefix + bytes([byte]) + b'"}'
                reply = db.raw(data)
                assert not reply["ok"] and reply["error"]["code"] == "protocol_error", reply
        db.close()
    print(f"PASS {binary}: 582 SQL/JSON string round trips; 1950 raw JSON control-byte rejections", flush=True)


def vectors(binaries):
    rng = random.Random(19260407)
    dimensions = (1, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 1024, 4096, 4097)
    checks = 0
    for dim in dimensions:
        # All tables exceed the sketch activation threshold. Quantization
        # limits, ties, long tails and scale changes all receive coverage.
        for scale in ([2.0**-135, 1.0, 2.0**120] if dim in (33, 128) else [1.0]):
            data = [[f32(rng.uniform(-1, 1) * scale) for _ in range(dim)] for _ in range(520)]
            data[0] = [0.0] * dim
            data[1] = data[2] = [f32(63 / 64 * scale)] * dim
            data[3] = [f32(-63 / 64 * scale)] * dim
            data[4] = data[5] = data[6] = list(data[7])
            queries = [data[0], data[1], list(data[7]),
                       [f32(rng.uniform(-1, 1) * scale) for _ in range(dim)],
                       [f32(2 * scale)] * dim]
            if dim == 128:
                # Nearly tied FP32 vectors: the integer filter must not decide
                # the ordering without returning to the original coordinates.
                data[8] = [f32(scale)] * dim
                data[9] = [f32(scale)] * dim
                data[9][-1] = f32((1 + 2.0**-23) * scale)
            expected = []
            sql_queries = []
            for qid, query in enumerate(queries):
                group = qid % 2
                candidates = range(520) if not group else range(1, 520, 3)
                truth = sorted((math.sqrt(sum((a-b)**2 for a,b in zip(data[i], query))), i)
                               for i in candidates)[:17]
                expected.append(truth)
                sql_queries.append(f"SELECT id,L2_DISTANCE(v,{literal(query)}) AS d FROM t" +
                                   (" WHERE g='g1'" if group else "") + " ORDER BY d LIMIT 17")
            previous = None
            for binary in binaries:
                with tempfile.TemporaryDirectory() as directory:
                    db = Session(binary, directory)
                    db.sql(f"CREATE TABLE t(id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR({dim}))")
                    for start in range(0, 520, 40):
                        db.sql("INSERT INTO t VALUES " + ",".join(
                            f"({i},'g{i%3}',{literal(data[i])})" for i in range(start, min(start+40, 520))))
                    db.sql("CREATE VECTOR INDEX ix ON t(v)")
                    results = [db.sql(sql) for sql in sql_queries]
                    for truth, rows in zip(expected, results):
                        assert [i for _, i in truth] == [r[0] for r in rows], (dim, scale, binary, truth, rows)
                        assert all(math.isclose(d, row[1], rel_tol=1e-12, abs_tol=1e-150)
                                   for (d, _), row in zip(truth, rows)), (dim, scale)
                    if previous is not None:
                        # FMA and scalar reductions can differ in final bits.
                        # Require identical IDs and a tighter distance tolerance
                        # than the public protocol's 1e-5.
                        for current_rows, prior_rows in zip(results, previous):
                            assert [r[0] for r in current_rows] == [r[0] for r in prior_rows]
                            assert all(math.isclose(a[1], b[1], rel_tol=1e-12, abs_tol=1e-150)
                                       for a, b in zip(current_rows, prior_rows))
                    previous = results
                    db.close()
                checks += len(sql_queries)
            print(f"PASS dimension={dim} scale={scale} exact IDs, ties, filters and distances", flush=True)
    print(f"PASS {checks} independent brute-force vector queries across {len(binaries)} executable(s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binaries", nargs="+")
    args = parser.parse_args()
    for binary in args.binaries:
        strings(binary)
    vectors(args.binaries)


if __name__ == "__main__":
    main()
