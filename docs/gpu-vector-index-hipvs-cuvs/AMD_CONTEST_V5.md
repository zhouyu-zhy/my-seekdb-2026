# AMD 赛道 V5 · CAGRA 快照优化

基线为 `634888d9b037153fa143d2c913ba4b25503e1d62`。候选保留 V5 的 NN-Descent 建图配置、FP32、L2 和 hipVS CAGRA 查询。改动覆盖 `CREATE VECTOR INDEX ... lib=cuvs` 的快照写入和首次重载。CPU `lib=vsag` 继续使用原序列化路径。

## 数据路径

V5 将行 ID、原始 FP32 向量和含完整设备向量副本的 hipVS 文件写入 `OBCUVS02`。读写都需要经临时文件转存 hipVS 文件。候选使用 `OBCUVS03`，从 GPU 导出原索引的邻接矩阵，按行写入 ID、FP32 向量和 `uint32_t` 邻居偏移。重载由 hipVS `cuvsCagraIndexFromArgs` 把两组数据恢复到 CAGRA 索引，转移完成后释放暂存邻接表。既有 `OBCUVS02` 的读取路径保留。构建、导出或重载失败时仍由调用方报错。

以 250,000 行、128 邻接度为例，新格式每行占 `8 + 4 × 维度 + 4 × 128` 字节，另有 32 字节头。在 128 和 256 维时分别为 258,000,032 和 386,000,032 字节。相对于旧格式，至少少写一份逻辑 FP32 向量副本，分别为 128,000,000 和 256,000,000 字节。以上是格式推导，实际耗时、Recall 和显存峰值需在 Radeon 实测。

## 编译与功能验收

在题面锁定的 ROCm 7.2.4 及 hipVS `f96a663d0a4441d6f5344bd3ef552b71ae7b370b` 环境下，对冻结基线和候选分别使用相同的赛题编译命令。仓库的 bridge 编译入口和 seekdb CUVS 选项保持原样。

```bash
tools/hipvs_bridge/build.sh /tmp/libseekdb_cuvs_bridge.so
g++ -std=c++17 -O2 tools/hipvs_bridge/test_persistence.cpp \
    -L/tmp -lseekdb_cuvs_bridge -Wl,-rpath,/tmp -o /tmp/cagra-persistence
/tmp/cagra-persistence graph
/tmp/cagra-persistence write /tmp/cagra-v2.bin
/tmp/cagra-persistence read /tmp/cagra-v2.bin
```

`graph` 比较导出前和重载后的 16 组 Top-100 ID 与距离，并在检索前释放原 GPU 索引、暂存邻接表和 FP32 输入。两条兼容性命令使用原有 hipVS 文件序列化。另可使用本目录现有的 [DDL 持久化验收脚本](validation/run_ddl_persistence_smoke.sh)，核对首条查询使用 GPU、进程重启后读取快照，以及同一实例内 CPU VSAG 仍可工作。

## CPU / GPU 构建对比

在同一台 AMD 评测环境中分别启动冻结 V5 和候选，使用相同的未建向量索引表、相同的查询向量及 mysql 客户端。以下是开发对比入口；正式评分仍由比赛平台执行。

```bash
# 表结构：CREATE TABLE items(id BIGINT PRIMARY KEY, v VECTOR(128));
# 将同一份 128 维数据预先导入两个实例；另对 256 维数据独立重复。
M=docs/gpu-vector-index-hipvs-cuvs/validation/measure_offline_build.py
python3 "$M" --database bench --table items --queries queries.jsonl \
  --port 2881 --library cuvs --label v5-cuvs > v5-cuvs.json
python3 "$M" --database bench --table items --queries queries.jsonl \
  --port 2882 --library cuvs --label candidate-cuvs > candidate-cuvs.json
python3 "$M" --database bench --table items --queries queries.jsonl \
  --port 2882 --library vsag --label candidate-vsag > candidate-vsag.json
```

脚本计时包含同一客户端调用内的离线 DDL 和首条合法查询；原始数据导入与精确结果计算在计时之外。每轮创建并清理独立索引，随后报告 Top-10 / Top-100 Recall 和单条 SQL 的整体吞吐。使用已计算的中位时间报告 CPU/GPU 对比，以及 V5 GPU/候选 GPU 的配对加速比；交换两版本运行顺序，分开记录两个维度。可结合 GPU profiler 核对正式路径及显存峰值。

## 本地验证记录

| 检查 | 结果 |
| --- | --- |
| Bridge 与回归程序编译 | GCC 13.3 使用固定版 hipVS C API 及 ROCm 7.2.4 头文件生成目标文件 |
| 主机侧快照协议 | 在替身 bridge 上通过 ASan/UBSan 验证 V3 往返、截断拒绝、长度校验和 V2 重载 |
| CPU 测量脚本试跑 | 已发布的 seekdb 1.4.0 Windows 包，4096 行、128 维、32 条查询、两轮 VSAG DDL 加首查分别为 1.970 s 和 1.862 s |
| 完整候选编译、Radeon 回归及性能 | 本机只有 NVIDIA RTX 4060，缺少 ROCm 编译环境和可用 Radeon 实例 |

CPU 测量只验证脚本能执行真实 SQL 和生成报告。候选 CPU/GPU 对比、V5 GPU/候选 GPU 加速比、Recall 变化与显存峰值均待同机实测。

[赛题与验收规则](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/seebnbaoxot1b0tb) · [AMD 部署说明](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/lcuknwgl955v8ou7)
