# MiniVectorDB

单机、单进程、单连接（同一时刻最多一个显式事务）的轻量向量数据库，实现
`2026-competition-round2` 要求的 SQL 子集、精确向量检索、事务与崩溃恢复。
本题包中的 `mysqltest/`、`protocol_check/`、`STDIO_PROTOCOL.md` 是组织方材料，
原样保留，未作修改。

## 1. 语言与运行环境

| 项目 | 说明 |
| --- | --- |
| 语言 | Python 3.12（镜像内为 Ubuntu 24.04 的 `python3=3.12.3-0ubuntu2.1`） |
| 依赖 | `python3-numpy=1:1.26.4+ds-6ubuntu1`（可选：缺省时自动退回纯 Python 扫描） |
| 数据目录 | `/data`（表数据、`catalog.json`、`commit.log` 全部在此） |
| 通信 | 标准输入/输出 JSON Lines，见 `STDIO_PROTOCOL.md`；日志只写 stderr |
| 端口 | 不监听任何网络端口 |

除组织方材料外的全部代码位于 `src/minivectordb/`：

```text
src/minivectordb/
├─ __main__.py    入口：python3 -m minivectordb --data-dir /data
├─ server.py      JSONL 协议、ready 握手、shutdown、错误映射
├─ sqlparser.py   手写词法/语法分析器（CREATE/INSERT/SELECT/UPDATE/DELETE/事务）
├─ engine.py      目录、事务、执行器、列式快路径、向量 TopK
├─ storage.py     分帧追加日志、CRC 校验、恢复、checkpoint 重写、catalog
├─ values.py      BIGINT / VARCHAR(n) / VECTOR(n) 取值规则与 float32 校验
└─ errors.py      错误码
```

## 2. 构建与启动

```bash
docker build -t minivectordb:latest .
docker run --rm -i -v minivectordb-data:/data --cpus=2 --memory=2g minivectordb:latest
```

镜像以 `ENTRYPOINT ["python3","-m","minivectordb"]`、`CMD ["--data-dir","/data"]`
前台启动，不分配 TTY、不暴露端口。启动后先完成恢复再输出 `{"type":"ready"}`。

## 3. 从空数据目录开始的完整示例

以下请求逐行写入 stdin，每行得到一行响应（`python3 -m minivectordb --data-dir /tmp/db`）：

```jsonl
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(32), content VARCHAR(1024), embedding VECTOR(4));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1,'database','事务与恢复','[1,0,0,0]'),(2,'database','向量索引','[0,1,0,0]'),(3,'travel','旅行攻略','[0,0,1,0]');","search_mode":"exact"}
{"id":3,"op":"execute","sql":"SELECT id, content, L2_DISTANCE(embedding,'[0.9,0.1,0,0]') AS distance FROM docs WHERE category = 'database' ORDER BY distance LIMIT 2;","search_mode":"exact"}
{"id":4,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":5,"op":"execute","sql":"UPDATE docs SET embedding = '[0.8,0.2,0,0]' WHERE id = 1;","search_mode":"exact"}
{"id":6,"op":"execute","sql":"DELETE FROM docs WHERE id = 2;","search_mode":"exact"}
{"id":7,"op":"execute","sql":"ROLLBACK;","search_mode":"exact"}
{"id":8,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding,'[0.9,0.1,0,0]') AS distance FROM docs ORDER BY distance;","search_mode":"exact"}
{"id":9,"op":"shutdown"}
```

响应（实际输出，命令：`python3 src/minivectordb/__main__.py --data-dir /tmp/demo-db < tools/demo.jsonl`）：

```text
{"type":"ready"}
{"id":1,"ok":true}
{"id":2,"ok":true,"affected_rows":3}
{"id":3,"ok":true,"columns":["id","content","distance"],"rows":[[1,"事务与恢复",0.141421377658844],[2,"向量索引",1.2727922201156616]]}
{"id":4,"ok":true}
{"id":5,"ok":true,"affected_rows":1}
{"id":6,"ok":true,"affected_rows":1}
{"id":7,"ok":true}
{"id":8,"ok":true,"columns":["id","distance"],"rows":[[1,0.141421377658844],[2,1.2727922201156616],[3,1.3490737676620483]]}
{"id":9,"ok":true}
```

`id=1` 的向量更新与 `id=2` 的删除在 `ROLLBACK` 后对第 8 条查询完全不可见：
距离仍是事务前的 0.141421 与 1.272792。

重启：再次以相同 `/data` 启动进程即可，恢复完成后同样先输出 `{"type":"ready"}`，
已提交数据保持一致；`--restart` 用例见 `mysqltest/cases/storage_restart.test`。

## 4. SQL 支持范围

| 语句 | 说明 |
| --- | --- |
| `CREATE TABLE t (...)` | `BIGINT`、`VARCHAR(n)`、`VECTOR(n)`；一个 `BIGINT` 主键（列级或表级）；每表最多一个向量列 |
| `CREATE VECTOR INDEX i ON t(c)` | 记录索引元数据；检索结果始终对应当前可见记录 |
| `INSERT INTO t VALUES (...),(...)` | 多行插入；可选列清单；主键重复返回 `duplicate_key` |
| `SELECT ... FROM t [WHERE ...] [ORDER BY ...] [LIMIT n]` | `*`、普通列投影、`L2_DISTANCE(col,'[...]') [AS alias]` |
| `UPDATE t SET ... [WHERE ...]` | 可同时更新多列；不允许改主键（返回 `unsupported_sql`） |
| `DELETE FROM t [WHERE ...]` | 无 `WHERE` 时删除全部可见行 |
| `BEGIN` / `COMMIT` / `ROLLBACK` | 同时接受 `START TRANSACTION`、`COMMIT WORK`、`ROLLBACK WORK` |

`WHERE` 支持 `=`、`<>`、`<`、`<=`、`>`、`>=` 与 `AND`（不支持 `OR`）。
`ORDER BY` 支持多列与 `ASC`/`DESC`，并列项一律按主键升序。

## 5. 错误处理

失败语句返回 `{"ok":false,"error":{"code":...,"message":...}}`，进程继续服务；
失败语句不留下部分修改，事务中先前成功的语句仍可提交或回滚。

| 场景 | `error.code` |
| --- | --- |
| 主键重复 | `duplicate_key` |
| 向量维度不符 | `dimension_mismatch` |
| NaN / Infinity / float32 溢出 / 非法文本 | `invalid_vector` |
| `VARCHAR` 超长 | `value_too_long` |
| 不支持的语句（`ALTER TABLE`、`OR`、嵌套事务等） | `unsupported_sql` |
| 语法错误 | `syntax_error` |
| 表不存在 / 列不存在 | `table_not_found` / `unknown_column` |

## 6. 自测

```bash
# 官方驱动直接拉起源码（无需构建镜像）
python3 tools/local_check.py --handshake-only --output results/handshake
python3 tools/local_check.py --output results/public

# 官方驱动对镜像
docker build -t minivectordb:dev .
python3 protocol_check/run.py --image minivectordb:dev --output results/public-image

# 公开题包未覆盖的部分：强杀恢复、10k×128 benchmark
python3 tools/crash_check.py
python3 tools/bench.py --label columnar
MVDB_DISABLE_COLUMNAR=1 python3 tools/bench.py --label rows
```

`tools/` 只是本地自测脚本，不参与判题。已记录的结果见 `DESIGN.md` 第 5 节。
