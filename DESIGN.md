# MiniVectorDB 设计与验证记录

## 1. 磁盘布局

`/data` 下只有三类文件：

```text
/data
├─ catalog.json        表定义、索引元数据、下一个表号（唯一需要原地更新的文件）
├─ commit.log          事务提交点：每条记录只有 9 字节（op + txn id）
└─ table_<id>.dat      每张表一个追加日志
```

`table_<id>.dat` 先写 5 字节魔数 `MVD1\n`，随后是定长帧：

```text
| crc32(4) | len(4) | payload(len) |
payload = op(1) | txn(8) | pk(8) | row bytes     # op 1 = upsert, 2 = delete
```

行内编码按列声明顺序排列：`BIGINT` 为 8 字节小端整数，`VARCHAR` 为 4 字节长度 +
UTF-8 字节，`VECTOR(n)` 为 `n` 个 float32。写入即编码为 float32，之后所有距离都
用存储后的值计算，因此 `16777217` 与 `16777216` 的距离为 0，符合 `vector_float32`
用例。

## 2. 向量索引方案

`CREATE VECTOR INDEX` 目前只持久化索引元数据，检索走两条路径：

* **精确**：对过滤后的候选集逐个计算欧氏距离，等价于暴力扫描，`exact` 模式恒返回真值；
* **近似（`ann`）**：复用同一条精确路径，Recall@K 恒为 1.0。

选择理由：题目对 `ann` 的要求是“返回 ID 有效、无重复、满足过滤、对应当前可见记录
且达到召回率门槛”，精确扫描天然满足；在 10k×128 规模下，把向量列构建成
`numpy` 列式快照后单次 TopK 约 4 ms，比自研 HNSW 的插入维护与召回风险更划算。
代价是数据规模继续增大时查询会线性变慢（见第 7 节）。

TopK 的并列处理：先取第 K 小的距离值，再把所有 `<= 该值` 的行纳入，按
`(距离, 主键)` 排序后截断。直接用 `argpartition` 截断会在边界上丢掉主键更小的并列行。

## 3. 事务处理

同一时刻最多一个显式事务。已提交状态是内存中的 `{pk -> (offset, length)}` 映射；
事务在此基础上维护 overlay：

* 语句执行时先写记录（不带 fsync），把 `pk -> 新 entry` 放进 overlay，同时记录 undo；
* 语句失败：按 undo 回退 overlay，已经落盘的记录变成不可达垃圾（不可见即等同未发生）；
* `ROLLBACK`：丢弃 overlay，已写记录同样成为垃圾，由 checkpoint 清理；
* 事务内查询：`visible()` 把 overlay 合并进已提交映射，因此读得到自己的修改，向量
  检索与普通查询共用同一份可见集。

DDL 在显式事务中按 MySQL 语义隐式提交（先提交当前事务再执行），避免“回滚掉建表”。

## 4. 提交持久化与恢复

提交顺序严格为：

1. 对本次事务写过的每个表文件 `flush()` + `fsync()`；
2. 向 `commit.log` 追加一条 `commit(txn)` 记录并 `fsync()`；
3. 把 overlay 合并进可见映射，返回成功响应。

因此“收到 COMMIT 成功”必然意味着数据与提交点都已落盘。崩溃只会落在三步之间：

* 步骤 1 之前或之中：`commit.log` 里没有该事务 → 恢复时忽略它的全部记录；
* 步骤 2 之后、响应之前：提交点已持久，恢复时该事务完整生效（题目允许）；
* 任何时刻都不会出现“部分记录生效”：可见性只由 `commit.log` 中的一条提交点决定。

恢复流程：

1. 读 `commit.log`，得到已提交事务集合与最大事务号；尾部 CRC 或长度不合法则截断；
2. 逐表扫描 `table_<id>.dat`，只应用 `txn == 0`（checkpoint 写入的已提交记录）或
   `txn ∈ 已提交集合` 的记录，重建 `pk -> entry`；
3. 若某表仍残留未提交事务的记录（含恢复途中再次被强杀的情况），重写该表为只含可见
   行的 checkpoint 文件（`os.replace` 原子替换），使事务号可以安全重新计数；
4. 完成上述步骤后才输出 `{"type":"ready"}`。

正常 `shutdown`：回滚未提交事务 → 逐表 checkpoint → 清空 `commit.log` → fsync。
checkpoint 先写 `table_<id>.dat.compact`、fsync，再原子 rename，崩溃只会在新旧版本之间二选一。

## 5. 内存与“大于内存预算”的数据

* 行数据不常驻：默认只用 32 MiB 的有界解码行缓存（`MVDB_ROW_CACHE_BUDGET`），
  淘汰后回落到 `pread` 逐行读取；可见性只依赖 `pk -> offset` 映射。
* 列式快照有 96 MiB 预算（`MVDB_COLUMNAR_BUDGET`），估算超预算时不构建，直接走逐行
  参考路径。因此大表只损失速度，不受正确性影响。
* `MVDB_DISABLE_COLUMNAR=1` 可强制逐行路径，用于两条路径对拍。

## 6. 验证记录（本地实测）

| 项目 | 命令 | 结果 |
| --- | --- | --- |
| 公开用例（镜像） | `python3 protocol_check/run.py --image minivectordb:dev --output ...` | 20/20 PASS |
| 公开用例（源码，列式路径） | `python3 tools/local_check.py --output results/np-columnar` | 20/20 PASS |
| 公开用例（源码，强制逐行） | `MVDB_DISABLE_COLUMNAR=1 python3 tools/local_check.py --output results/np-rows` | 20/20 PASS |
| 握手 | `protocol_check/run.py --handshake-only` | PASS |
| 强杀恢复 | `python3 tools/crash_check.py` | 9/9 PASS（已确认提交保留、未提交丢弃、COMMIT 竞态无部分生效、恢复途中再杀可重复恢复） |
| 补充规范检查 | `python3 tools/spec_check.py` | 15/15 PASS（非法向量、表级主键、`SELECT *`、语句原子性、多语句请求、撕裂尾部恢复等） |
| 基准（列式） | `python3 tools/bench.py --label columnar` | 装载 1.99 s / 查询 5.07 ms per query / Recall@10 1.0 / 峰值内存 102 MiB / 数据 5.3 MB |
| 基准（逐行） | `MVDB_DISABLE_COLUMNAR=1 python3 tools/bench.py --label rows` | 装载 1.94 s / 查询 48.66 ms per query / Recall@10 1.0 / 峰值内存 88 MiB |

同一台机器上重复运行时查询耗时有 ±15% 波动（例如列式路径在 4.15–5.07 ms 之间），
上表是最后一次完整采集的值，比较结论只使用同一批次的两组数据。

开发过程中被这些验证抓到的真实缺陷：

1. 镜像内（有 numpy）空表投影报 `internal_error`：列式快照对空表没有为列建立数组，
   `KeyError(0)`；本地无 numpy 时走逐行路径所以看不出来 → 为空表预建空数组。
2. 强杀后已提交行丢失：`storage.recover()` 仍按“提交点写在表文件里”的旧设计实现，
   没有读取 `commit.log`，把所有带事务号的记录都当成孤儿 → 恢复必须先读提交点再建可见集。
3. 快照缓存以“事务对象身份”为键，隐式事务每条语句都是新对象，缓存永不命中，
   查询从 6.0 ms 退化到 41.4 ms → 改用每次写入自增的 generation 作为键。
4. 词法分析器逐字符扫描字符串字面量，128 维向量文本约 1.2 KB，装载 10k 行耗时
   2.94 s → 改用 `str.find` 定位结束引号后降到 1.86 s。

## 7. 已实现与已知限制

已实现：表定义与持久化、多行 INSERT、条件 UPDATE/DELETE、列投影、`*`、六种比较、
`AND`、多列 `ORDER BY` + `LIMIT`、`L2_DISTANCE` 精确 TopK 与带过滤 TopK、
`CREATE VECTOR INDEX`、`BEGIN/COMMIT/ROLLBACK`、自动提交、语句级原子性、强杀恢复。

已知限制：

* 无近似索引结构，`ann` 模式即精确扫描，规模增大时查询线性变慢；
* 无 `NULL`、`JOIN`、聚合、子查询、`ALTER TABLE`、嵌套事务、主键更新（均按题目不要求，
  统一返回 `unsupported_sql`，不做静默降级）；
* `ORDER BY` 字符串列 `DESC` 会退回逐行路径（列式路径只向量化数值列的降序）；
* 未实现多客户端并发：单进程单连接，符合题目限定范围；
* 事务写放大：更新即追加新版本，依赖 checkpoint 回收，长期高频更新需要更激进的
  压缩策略（当前仅在 shutdown 与“存在孤儿记录”时重写）。
