#!/usr/bin/env python3
"""Contest-shaped benchmark for the seekdb vector index build.

It reproduces the measurement the judges use:
  * load the vectors into a plain table (no index),
  * time `CREATE VECTOR INDEX ... WITH (distance=l2, type=hnsw, lib=<cuvs|vsag>,
    m=16, ef_construction=200, ef_search=64)` **plus the first fixed query**
    (the lazy-build guard of the rules),
  * run nq single-row approximate queries for Top-10 / Top-100, report QPS,
    Recall@10 / Recall@100 against an exact brute-force ground truth,
  * sample the GPU VRAM peak of the whole process from sysfs.

    python bench.py --x /root/data/sift250k_x.f32 --q /root/data/sift250k_q.f32 \
        --dim 128 --rows 250000 --nq 4096 --lib cuvs --tag candidate

Only the mysql command line client and numpy are required.
"""
import argparse
import json
import os
import subprocess
import threading
import time

import numpy as np

MYSQL = ['mysql', '-h', '127.0.0.1', '-P', '2881', '-uroot']
DB = 'vecbench'
TABLE = 'items'


def sql(stmt, db=DB):
    return subprocess.run(MYSQL + (['-D', db] if db else []), input=stmt,
                          capture_output=True, text=True)


def sql_one(stmt, db=DB):
    r = subprocess.run(MYSQL + ['-N', '-s'] + (['-D', db] if db else []),
                       input=stmt, capture_output=True, text=True)
    return r.stdout.strip()


def vram_used():
    for card in sorted(os.listdir('/sys/class/drm')):
        p = '/sys/class/drm/%s/device/mem_info_vram_used' % card
        if card.startswith('card') and os.path.exists(p):
            try:
                v = int(open(p).read().strip())
            except Exception:
                continue
            if v > 64 << 20:
                return v
    return 0


class VramSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop = False

    def run(self):
        while not self._stop:
            self.peak = max(self.peak, vram_used())
            time.sleep(self.interval)

    def stop(self):
        self._stop = True


def load_rows(x, dim):
    sql('CREATE DATABASE IF NOT EXISTS %s;' % DB, None)
    sql('DROP TABLE IF EXISTS %s;' % TABLE)
    sql('CREATE TABLE %s(id INT PRIMARY KEY, v VECTOR(%d));' % (TABLE, dim))
    t0 = time.time()
    batch = 500
    for s in range(0, len(x), batch):
        e = min(s + batch, len(x))
        rows = ','.join("(%d,'[%s]')" % (i, ','.join('%.6g' % v for v in x[i]))
                        for i in range(s, e))
        r = sql('INSERT INTO %s(id,v) VALUES %s;' % (TABLE, rows))
        if r.returncode != 0:
            raise SystemExit('insert failed at %d: %s' % (s, r.stderr[:300]))
    return time.time() - t0


def exact_topk(x, q, k):
    out = np.empty((len(q), k), dtype=np.int64)
    xt = x.T.copy()
    xn = (x * x).sum(axis=1)
    step = 32
    for s in range(0, len(q), step):
        e = min(s + step, len(q))
        d = q[s:e] @ xt
        d *= -2.0
        d += xn
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        for r in range(e - s):
            idx = part[r]
            out[s + r] = idx[np.argsort(d[r, idx])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', required=True)
    ap.add_argument('--q', required=True)
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--rows', type=int, default=250000)
    ap.add_argument('--nq', type=int, default=4096)
    ap.add_argument('--lib', default='cuvs', choices=['cuvs', 'vsag'])
    ap.add_argument('--tag', default='run')
    ap.add_argument('--skip-load', action='store_true')
    ap.add_argument('--skip-query', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    x = np.fromfile(args.x, dtype=np.float32).reshape(args.rows, args.dim)
    q = np.fromfile(args.q, dtype=np.float32).reshape(args.nq, args.dim)

    sampler = VramSampler()
    sampler.start()
    load_s = None
    if not args.skip_load:
        load_s = load_rows(x, args.dim)

    sql('DROP INDEX idx ON %s;' % TABLE) if sql_one(
        "SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema='%s' "
        "AND table_name='%s' AND index_name='idx';" % (DB, TABLE)) != '0' else None

    sampler.peak = 0
    t0 = time.time()
    stmt = ("CREATE VECTOR INDEX idx ON %s(v) WITH (distance=l2, type=hnsw, "
            "lib=%s, m=16, ef_construction=200, ef_search=64);" % (TABLE, args.lib))
    r = sql(stmt)
    if r.returncode != 0:
        raise SystemExit('create index failed: %s' % r.stderr[:500])
    first = sql_one("SELECT id FROM %s ORDER BY l2_distance(v,'[%s]') APPROXIMATE LIMIT 100;"
                    % (TABLE, ','.join('%.6g' % v for v in q[0])))
    build_s = time.time() - t0
    peak_build = sampler.peak

    result = {'tag': args.tag, 'lib': args.lib, 'dim': args.dim, 'rows': args.rows,
              'nq': args.nq, 'load_s': load_s, 'build_s': round(build_s, 3),
              'vram_peak_MiB': round(peak_build / 1048576, 1),
              'first_query_rows': len(first.split())}

    if not args.skip_query:
        gt = {k: exact_topk(x, q, k) for k in (10, 100)}
        for k in (10, 100):
            t1 = time.time()
            hits = 0
            for i in range(len(q)):
                ids = sql_one("SELECT id FROM %s ORDER BY l2_distance(v,'[%s]') "
                              "APPROXIMATE LIMIT %d;" % (TABLE, ','.join('%.6g' % v for v in q[i]), k))
                got = [int(v) for v in ids.split()] if ids else []
                if len(set(got)) != k:
                    raise SystemExit('result_integrity: top-%d returned %d ids' % (k, len(set(got))))
                hits += len(set(got) & set(gt[k][i].tolist()))
            qps = len(q) / (time.time() - t1)
            result['qps_top%d' % k] = round(qps, 1)
            result['recall@%d' % k] = round(hits / float(len(q) * k), 5)

    sampler.stop()
    result['vram_peak_MiB'] = round(max(result['vram_peak_MiB'], sampler.peak / 1048576), 1)
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
