# MiniVectorDB

面向 OceanBase 2026 加赛题目的最小向量数据库实现。实现语言为 C++17，单进程运行，通过 stdin/stdout 的 JSON Lines 协议与官方驱动交互，所有数据只写入 `/data`。

## 1. 快速开始

### 本地构建

```bash
cd minivectordb
make                # 同时生成通用版 minivectordb 和 AVX2 版 minivectordb-avx2
./minivectordb --data-dir /tmp/mvdb
```

`make generic` / `make avx2` 可单独构建。AVX2 版只用于距离计算密集的查询路径，入口脚本会在 CPU 不支持 AVX2+FMA 时自动回退到通用版。

### Docker 构建与运行

在仓库根目录：

```bash
docker build -t minivectordb:local .
docker run --rm -i -v /tmp/mvdb-data:/data minivectordb:local
```

镜像入口为 `/app/docker-entrypoint.sh`，默认参数 `--data-dir /data`；入口脚本只读取 `/proc/cpuinfo` 做一次 CPU 能力判断，不写 `/data` 以外的位置。

### 协议示例

启动后第一行输出：

```json
{"type":"ready"}
```

之后每行一个请求，每个请求返回一行响应：

```json
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(32), embedding VECTOR(4));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1,'database','[0.1,0.2,0.3,0.4]'),(2,'database','[0.2,0.1,0.4,0.3]');"}
{"id":3,"op":"execute","sql":"SELECT id, L2_DISTANCE(embedding,'[0,0,0,0]') AS distance FROM docs WHERE category = 'database' ORDER BY distance LIMIT 10;","search_mode":"exact"}
{"id":4,"op":"execute","sql":"BEGIN;"}
{"id":5,"op":"execute","sql":"UPDATE docs SET embedding = '[0.8,0.2,0,0]' WHERE id = 1;"}
{"id":6,"op":"execute","sql":"ROLLBACK;"}
{"id":7,"op":"shutdown"}
```

重启只需复用同一个数据目录：

```bash
printf '%s\n' \
  '{"id":1,"op":"execute","sql":"SELECT id, category FROM docs ORDER BY id;"}' \
  '{"id":2,"op":"shutdown"}' \
  | ./minivectordb --data-dir /tmp/mvdb
```

## 2. SQL 支持范围

- `CREATE TABLE`：`BIGINT`、`VARCHAR(n)`、`VECTOR(n)`；必须有且只有一个 `BIGINT PRIMARY KEY`；最多一个向量列。
- `INSERT`：单行/多行 `VALUES`，支持可选列清单。
- `SELECT`：列投影、`*`、`WHERE` 的 `=` `<>` `!=` `<` `<=` `>` `>=` 与 `AND`（支持括号）、`ORDER BY`（多键、`ASC`/`DESC`）、`LIMIT`。
- 向量函数：`L2_DISTANCE(vector_column, '[...]')`，支持 `AS` 别名。
- `UPDATE` / `DELETE`：按主键或条件，`UPDATE` 支持列 = 字面量列表。
- 事务：`BEGIN` / `START TRANSACTION`、`COMMIT`、`ROLLBACK`；DDL 也在事务内生效，可回滚。
- `CREATE VECTOR INDEX index_name ON table(vector_column)`：记录索引元数据并参与持久化恢复。
- 未实现：`NULL`、`JOIN`、聚合、子查询、`GROUP BY`、`OR`、`ALTER TABLE`、`DROP TABLE`、外键、主键更新、嵌套事务、`LIMIT ... OFFSET`。这些语句返回错误，不会静默改变语义。

主要错误码：`syntax_error`、`unsupported_sql`、`table_not_found`、`table_exists`、`column_not_found`、`duplicate_key`、`dimension_mismatch`、`invalid_vector`、`value_too_long`、`invalid_number`、`type_mismatch`、`invalid_schema`、`index_exists`、`transaction_active`、`no_transaction`。

## 3. 数据目录与持久化

`--data-dir` 下只有两类常驻文件：

- `catalog.dat`：当前 epoch 与活动数据文件名。
- `rows_N.dat`：数据文件，同时承担 WAL 与 checkpoint 后的基线数据。

一个提交对应文件尾部一个事务块：

```text
[magic 4B][version 4B][txn_id 8B][payload_length 8B][payload_crc 4B][payload][payload_length 8B]
```

- payload 内是若干 `CREATE TABLE` / `UPSERT` / `DELETE` / `CREATE VECTOR INDEX` 帧。
- payload 完整编码后才写块头、payload 和块尾；写完后 `fdatasync`（Windows 为 `_commit`），再在内存中应用并回复 `ok`。
- 启动时从文件头顺序扫描，校验 magic、长度上界和 CRC；遇到撕裂块或非法块立即截断到最后一个完整块，因此未确认的提交只会表现为“完整存在”或“完全不存在”，不会出现部分行。
- 提交后按阈值做 checkpoint：把当前所有表重写为新的 `rows_{epoch+1}.dat`，`fsync` 后原子替换 `catalog.dat`，再删除旧文件；崩溃时要么旧文件生效，要么新文件生效。
- 正常 `shutdown` 只清理未提交事务，不丢弃已提交数据；force kill 后由上述日志恢复。

## 4. 事务与可见性

- 每个会话只有一个活动事务。显式事务中的修改先进入内存 overlay，查询会把 overlay 覆盖到基线数据上，因此支持 read-your-writes。
- `COMMIT` 将 overlay 编码为一个事务块并持久化，然后应用到内存；`ROLLBACK` 直接丢弃 overlay。
- 自动提交语句也走同一个事务块路径，失败时不会留下部分行。
- 向量检索和普通扫描使用同一份可见性视图，所以事务内更新/删除后查询结果一致，回滚后恢复原结果。

## 5. 向量检索

- 向量按 `float32` 存储；L2 距离用 `double` 累加，最后开方，避免大分量时 float 溢出，同时保持与 float32 存储语义一致。
- 精确 TopK 使用固定大小最大堆：按距离升序，距离相同按主键升序；过滤先于距离计算，不会先取全表 TopK 再过滤。
- 有索引时 `search_mode: "exact"` 仍走精确扫描；`"ann"` 也返回精确结果。这是有意为之：题目把 ANN Recall@10 作为硬指标，而 10k 级别数据上精确扫描的资源开销可控，优先保证正确性和召回率，避免为索引维护引入不一致风险。`CREATE VECTOR INDEX` 的元数据会持久化，重启后仍可识别。
- 对已缓存的表，首次使用等值过滤的 TopK 时建立该列的有序行位次表，通过二分查找定位候选行，再验证全部谓词。写入会使位次表失效，事务覆盖层走可见行扫描。
- AVX2 路径为维度适中的缓存表生成一字节每维的向量摘要。缩放和舍入的误差上界用于拒绝候选，接近边界、不能安全量化及最终入围的行按原始向量和双精度距离计算。通用构建直接计算双精度距离。

## 6. 大数据与内存

- 根据 cgroup `memory.max` 推导缓存预算（默认上限 256MB，下限 8MB），只缓存行实体。
- 当数据文件大于缓存预算时，表进入流式模式：内存只保留 `pk -> (offset, size)` 的轻量引用，查询按块扫描文件并只解码命中行；因此数据量可以超过内存限额。
- checkpoint 按 4MB payload 分块写新文件，避免一次性构造全量镜像。
- 未缓存表的 TopK 使用引用表 + 流式扫描；过滤仍先于距离计算。

## 7. 本地验证记录

以下为开发机（WSL2，AVX2+FMA，数据目录在本地 ext4）上的自测结果，不代表评测机绝对数值：

- 官方公开用例：20/20 PASS（握手 + sql/storage/vector/transaction 全部通过）。
- 10,000 行 × 128 维、1,000 次 TopK（K=10，一半带 `category` 过滤）：启动约 0.01s，写入约 0.72–0.79s，1,000 次查询约 0.39–0.57s（无过滤与有过滤分别计时），峰值 RSS 约 13–18MB。
- 120,000 行 × 32 维写入后 kill -9 再启动：最后已确认提交的全部行可见；在提交块写入过程中 kill，重启后只看到“之前已提交的 1 行”，没有部分批次。
- 80,000 行数据在启动恢复过程中连续 kill：再次启动仍可完整恢复。
- 20,000 行 × 32 维反复更新触发 checkpoint：活动文件从 `rows_0.dat` 切换到 `rows_2.dat`，重启后更新、删除、插入和索引元数据均正确。

本次增量复验使用真实 Linux amd64 Docker 镜像和官方 `protocol_check/run.py`，全部 20 个公开用例通过。还用 `tests/check_engine.py` 对照未优化提交 `bbbd63957d42d2ebf7ec7ddfad9283141a50b007` 的 AVX2 可执行文件，使用独立随机数据、100 个持久化写入批次及 1,000 条查询交替执行三轮。最终核对包括随机精确 TopK、独立暴力真值、极值、索引缓存失效、事务失败、强杀、损坏日志尾部，以及恢复过程中再次强杀。

```bash
python3 minivectordb/tests/check_engine.py --baseline /path/to/baseline --candidate minivectordb/minivectordb-avx2
```

## 8. 已知限制

- 单客户端、串行执行；未实现多连接并发控制。
- 大事务（例如一次性 UPDATE/DELETE 全表）的 overlay 仍在内存中构建，极端情况下会受内存限制；正常批量写入和流式查询不受影响。
- 未实现真正的 IVF/HNSW 索引；`ann` 模式以精确扫描满足召回率要求，后续可在现有 `TableData` 向量缓存之上接入 IVF 列表并保留精确回退。
- 不支持 `NULL`、表达式更新、子查询和聚合；不支持在事务内嵌套事务。
- 崩溃恢复假设数据文件只由本程序写入；不处理外部篡改或磁盘写失败后的继续服务。

## 9. 文件结构

```text
minivectordb/
├── Makefile
├── docker-entrypoint.sh
├── src/
│   ├── mvdb.h        # 公共类型、协议结果、Engine 声明
│   ├── main.cpp      # JSON Lines 协议、请求分发、响应输出
│   ├── parser.cpp    # 词法/语法解析、字面量与向量解析
│   ├── storage.cpp   # 文件格式、事务块、恢复、checkpoint、缓存
│   └── engine.cpp    # 执行器、事务 overlay、查询/排序/TopK
└── README.md
```
