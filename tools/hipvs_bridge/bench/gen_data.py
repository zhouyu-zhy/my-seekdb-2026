#!/usr/bin/env python3
"""Generate the two 250K benchmark datasets used by the contest.

SIFT250K  : 250K x 128, taken from the public SIFT1M fvecs files.
HD256     : 250K x 256, the second contest workload is hidden; for local
            development this script builds a 256-dim proxy by concatenating the
            SIFT1M base descriptors in two different random orders, which keeps
            the structure of a real descriptor corpus while doubling the
            dimension.

Output: two float32 row-major raw files (<prefix>_x.f32 / <prefix>_q.f32)
which is exactly what `load_data.py` consumes.

    python gen_data.py --sift-dir /root/sift --out-dir /root/data
"""
import argparse
import os

import numpy as np


def read_fvecs(path, n=None):
    with open(path, 'rb') as f:
        dim = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        f.seek(0)
        total = os.path.getsize(path) // (4 * (dim + 1))
        n = total if n is None or n > total else n
        raw = np.fromfile(f, dtype=np.int32, count=n * (dim + 1)).reshape(n, dim + 1)
    return np.ascontiguousarray(raw[:, 1:]).view(np.float32), dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sift-dir', default='/root/sift')
    ap.add_argument('--out-dir', default='/root/data')
    ap.add_argument('--rows', type=int, default=250000)
    ap.add_argument('--nq', type=int, default=4096)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base, dim = read_fvecs(os.path.join(args.sift_dir, 'sift_base.fvecs'), args.rows)
    qry, _ = read_fvecs(os.path.join(args.sift_dir, 'sift_query.fvecs'), args.nq)

    base.astype(np.float32).tofile(os.path.join(args.out_dir, 'sift250k_x.f32'))
    qry.astype(np.float32).tofile(os.path.join(args.out_dir, 'sift250k_q.f32'))
    print('sift250k', base.shape, qry.shape)

    rs = np.random.RandomState(42)
    half = args.rows // 2
    a = base[:half]
    b = base[half:2 * half][rs.permutation(half)]
    hd = np.concatenate([a, b], axis=1).astype(np.float32)
    rs.shuffle(hd)
    qa = qry[:, :]
    qh = np.concatenate([qa, qa[rs.permutation(len(qa))]], axis=1).astype(np.float32)
    hd.tofile(os.path.join(args.out_dir, 'hd256_x.f32'))
    qh.tofile(os.path.join(args.out_dir, 'hd256_q.f32'))
    print('hd256', hd.shape, qh.shape)


if __name__ == '__main__':
    main()
