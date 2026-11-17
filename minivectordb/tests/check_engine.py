import argparse, json, math, os, random, resource, struct, subprocess, tempfile, time
from pathlib import Path

class DB:
    def __init__(self, binary, directory, cap=False):
        env = dict(os.environ)
        if cap:
            env["MVDB_CACHE_MB"] = "8"
        def limits():
            if cap:
                lim = 64 * 1024**2
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        self.p = subprocess.Popen([binary,"--data-dir",directory],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env,preexec_fn=limits if cap else None)
        assert json.loads(self.p.stdout.readline())["type"]=="ready"
        self.id=0
    def exec(self, sql, ok=True, mode="exact"):
        self.id+=1
        self.p.stdin.write(json.dumps({"id":self.id,"op":"execute","sql":sql,"search_mode":mode},ensure_ascii=False)+"\n")
        self.p.stdin.flush()
        result=json.loads(self.p.stdout.readline())
        assert result["id"]==self.id and result["ok"]==ok,(sql[:256],result)
        return result.get("rows",[])
    def end(self):
        self.p.stdin.write(json.dumps({"id":self.id+1,"op":"shutdown"})+"\n")
        self.p.stdin.flush()
        assert json.loads(self.p.stdout.readline())["ok"]
        assert self.p.wait(timeout=15)==0
    def kill(self):
        self.p.kill()
        assert self.p.wait(timeout=15)==-9
    def rss(self):
        return next(int(s.split()[1]) for s in Path(f"/proc/{self.p.pid}/status").read_text().splitlines() if s.startswith("VmHWM:"))

parser = argparse.ArgumentParser(description="Run MiniVectorDB performance and recovery checks")
parser.add_argument("--baseline", type=lambda x: str(Path(x).resolve()), required=True)
parser.add_argument("--candidate", type=lambda x: str(Path(x).resolve()), required=True)
args = parser.parse_args()
before, after = args.baseline, args.candidate
rng=random.Random(920026)
emit=lambda a: "'["+','.join(map(str,a))+"]'"
def f32(v): return struct.unpack("f",struct.pack("f",v))[0]
vecs=[[round(rng.uniform(-1,1),7) for _ in range(128)] for _ in range(10000)]
q=[[round(rng.uniform(-1,1),7) for _ in range(128)] for _ in range(1000)]
writes=["INSERT INTO d VALUES "+','.join(f"({i},'g{i%7}',{emit(vecs[i])})" for i in range(j,j+100)) for j in range(0,10000,100)]
def query(j, vector):
    where=f" WHERE g = 'g{j%7}'" if j%2 else ""
    return f"SELECT id,L2_DISTANCE(v,{emit(vector)}) AS dist FROM d{where} ORDER BY dist LIMIT 10"
queries=[query(j,v) for j,v in enumerate(q)]
results={}
for r in range(3):
    for binary in ([before,after] if r%2==0 else [after,before]):
        with tempfile.TemporaryDirectory() as data:
            db=DB(binary,data)
            db.exec("CREATE TABLE d (id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR(128))")
            t=time.monotonic()
            for sql in writes: db.exec(sql)
            write=time.monotonic()-t
            db.exec("CREATE VECTOR INDEX ix ON d(v)")
            db.exec(queries[0])
            t=time.monotonic()
            rows=[db.exec(sql) for sql in queries]
            query_time=time.monotonic()-t
            if r==0 and binary==before:
                results["rows"]=rows
            else:
                assert rows==results["rows"],"query response differs from unoptimized database"
            peak=db.rss()
            db.end()
            print(Path(binary).name,r+1,f"write={write:.4f}","query="+f"{query_time:.4f}","rss_kib="+str(peak),flush=True)
for j in range(20):
    qq=list(map(f32,q[j]))
    truth=sorted((math.sqrt(sum((f32(a)-b)**2 for a,b in zip(v,qq))),i) for i,v in enumerate(vecs) if not j%2 or i%7==j%7)[:10]
    got=results["rows"][j]
    assert len(got)==len(truth)
    for (dist,idx),(actual_id,actual_dist) in zip(truth,got):
        assert idx==actual_id and math.isclose(dist,actual_dist,rel_tol=1e-5)
print("PASS 3000 exact response comparisons across 3 paired runs and 20 independent brute-force queries",flush=True)
with tempfile.TemporaryDirectory() as data:
    db=DB(after,data)
    db.exec("CREATE TABLE d (id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR(33))")
    vectors=[[0]*33,[1]*33,[-1]*33,[1e38]*33,[-1e38]*33,[1e-44]*33]
    db.exec("INSERT INTO d VALUES "+','.join(f"({i},'a',{emit(v)})" for i,v in enumerate(vectors)))
    sql=f"SELECT id,L2_DISTANCE(v,{emit([0]*33)}) AS dist FROM d ORDER BY dist LIMIT 4"
    assert [r[0] for r in db.exec(sql)]==[0,5,1,2]
    assert [r[0] for r in db.exec(sql,mode="ann")]==[0,5,1,2]
    for bad in ("NaN","Infinity","9e38","1e999","1e-999"):
        db.exec("INSERT INTO d VALUES (99,'a','["+bad+", "+",".join(["0"]*32)+"]')",False)
    db.exec("BEGIN"); db.exec("UPDATE d SET v="+emit([0]*33)+" WHERE id=1")
    db.exec("INSERT INTO d VALUES (50,'a',"+emit([0]*33)+"),(0,'a',"+emit([0]*33)+")",False)
    assert db.exec("SELECT id FROM d WHERE id=50")==[]
    assert [r[0] for r in db.exec(sql)]==[0,1,5,2]
    db.exec("COMMIT")
    db.exec("BEGIN");db.exec("DELETE FROM d WHERE id=0");db.kill()
    db=DB(after,data)
    assert [r[0] for r in db.exec(sql)]==[0,1,5,2]
    db.end()
    wal=next(Path(data).glob("rows_*.dat"))
    with wal.open("ab") as f: f.write(b"torn partial log record")
    db=DB(after,data)
    assert [r[0] for r in db.exec(sql)]==[0,1,5,2]
    db.end()
print("PASS exact/ANN ties, ±1e38/denormals, illegal vectors, failed transaction statement, SIGKILL and torn tail",flush=True)
# More than 512 cached rows activates the quantized path. Change its scale
# and row ordinals with updates, deletes, and a rollback on the same table.
with tempfile.TemporaryDirectory() as data:
    db = DB(after, data)
    db.exec("CREATE TABLE d (id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR(33))")
    vectors_by_id = {i: [f32((i % 40) / 32)] * 33 for i in range(608)}
    for start in range(0, 608, 100):
        values = [f"({i},'g{i%2}',{emit(vectors_by_id[i])})" for i in range(start, min(start + 100, 608))]
        db.exec("INSERT INTO d VALUES " + ",".join(values))

    def check_topk(q0, group, mode):
        q = [f32(q0)] * 33
        sql = f"SELECT id,L2_DISTANCE(v,{emit(q)}) AS dist FROM d WHERE g='g{group}' ORDER BY dist LIMIT 17"
        expected = sorted((math.sqrt(sum((v - x)**2 for v, x in zip(vec, q))), i)
                          for i, vec in vectors_by_id.items() if i % 2 == group)[:17]
        actual = db.exec(sql, mode=mode)
        assert [i for _, i in expected] == [i for i, _ in actual]
        assert all(math.isclose(d, x[1], rel_tol=1e-5) for (d, _), x in zip(expected, actual))

    check_topk(0.5, 0, "exact")
    check_topk(1e38, 1, "exact")
    db.exec("BEGIN")
    db.exec("UPDATE d SET v=" + emit([1e38]*33) + " WHERE id=0")
    db.exec("DELETE FROM d WHERE id=2")
    db.exec("ROLLBACK")
    check_topk(0.5, 0, "ann")
    db.exec("UPDATE d SET v=" + emit([1e38]*33) + " WHERE id=0")
    vectors_by_id[0] = [f32(1e38)] * 33
    db.exec("DELETE FROM d WHERE id=2")
    del vectors_by_id[2]
    check_topk(0.5, 0, "exact")
    check_topk(1e38, 0, "ann")
    db.end()
print("PASS 608-row sketch ties, out-of-range query fallback, rollback, and committed index invalidation", flush=True)
with tempfile.TemporaryDirectory() as data:
    db=DB(after,data,True)
    db.exec("CREATE TABLE d (id BIGINT PRIMARY KEY,g VARCHAR(8),v VECTOR(768))")
    vector=emit([1]*768)
    for batch in range(245):
        db.exec("INSERT INTO d VALUES "+','.join(f"({i},'large',{vector})" for i in range(batch*100,(batch+1)*100)))
    wal=next(Path(data).glob("rows_*.dat"))
    assert wal.stat().st_size >64*1024**2
    db.kill()
    db=DB(after,data,True)
    assert db.exec("SELECT id FROM d WHERE id >= 24497 ORDER BY id")==[[24497],[24498],[24499]]
    sql=f"SELECT id,L2_DISTANCE(v,{vector}) AS dist FROM d WHERE g = 'large' ORDER BY dist LIMIT 3"
    assert db.exec(sql)==[[0,0],[1,0],[2,0]]
    db.end()
    # Kill a fresh startup while it holds the WAL open, before it reports ready.
    interrupted = subprocess.Popen([after,"--data-dir",data],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   env={**os.environ,"MVDB_CACHE_MB":"8"})
    try:
        import select
        deadline=time.monotonic()+5
        while time.monotonic() < deadline:
            fds=Path(f"/proc/{interrupted.pid}/fd")
            if any(f.resolve()==wal for f in fds.iterdir()):
                assert not select.select([interrupted.stdout],[],[],0)[0], "recovery already completed"
                break
            time.sleep(0.001)
        else:
            raise AssertionError("did not observe recovery opening the WAL")
    finally:
        interrupted.kill()
        assert interrupted.wait(timeout=15)==-9
    db=DB(after,data,True)
    assert db.exec("SELECT id FROM d WHERE id >= 24497 ORDER BY id")==[[24497],[24498],[24499]]
    print(f"PASS interrupted recovery, WAL={wal.stat().st_size} bytes under a 64 MiB address-space cap; peak RSS={db.rss()} KiB",flush=True)
    db.end()


# A candidate and its baseline should reopen each other's committed WAL. Run
# the same statements in separate directories to compare the wire bytes too.
with tempfile.TemporaryDirectory() as old_store, tempfile.TemporaryDirectory() as new_store:
    def write_fixture(binary, path):
        db = DB(binary, path)
        db.exec("BEGIN")
        db.exec("CREATE TABLE a (id BIGINT PRIMARY KEY, g VARCHAR(8), v VECTOR(3))")
        db.exec("CREATE TABLE b (id BIGINT PRIMARY KEY, g VARCHAR(8), v VECTOR(3))")
        db.exec("INSERT INTO b VALUES (8,'中','[1,-2,3]'),(2,'x','[-0.0,2e-40,1e38]')")
        db.exec("INSERT INTO a VALUES (1,'a','[1,2,3]'),(4,'b','[4,5,6]')")
        db.exec("UPDATE a SET v='[0,0,0]' WHERE id=4")
        db.exec("DELETE FROM b WHERE id=8")
        db.exec("CREATE VECTOR INDEX ai ON a(v)")
        db.exec("COMMIT")
        db.kill()
        return next(Path(path).glob("rows_*.dat")).read_bytes()

    assert write_fixture(before, old_store) == write_fixture(after, new_store)
    for binary, path in [(before, new_store), (after, old_store)]:
        db = DB(binary, path)
        assert db.exec("SELECT id,L2_DISTANCE(v,'[0,0,0]') AS d FROM a ORDER BY d LIMIT 1") == [[4, 0]]
        assert db.exec("SELECT id,g FROM b ORDER BY id") == [[2, "x"]]
        db.end()
print("PASS identical WAL bytes and bidirectional old/new recovery after SIGKILL", flush=True)
