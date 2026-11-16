# AMD 赛道 V5

本版本采用 FP32/L2 CAGRA、OBCUVS02 快照和原生 hipVS 序列化与重载接口。快照的临时文件复制按 1 MiB 分块，降低大图持久化时的读写调用次数。搜索复用索引内的设备临时内存和搜索参数对象，为查询向量、ID 和距离分别保留 256 字节对齐的地址。每个索引最多保留 1 MiB 设备缓冲区，大批量请求分配临时缓冲区并在完成后释放。索引内互斥锁保护并发搜索，失败路径同步设备流后才复用内存。距离复制失败时返回错误。单行近似查询同步提交到复用的 32 MiB 栈 pthread，减少每条 SQL 创建和销毁线程的开销。提交者等待执行完成后才释放查询缓冲区和索引引用。

使用比赛规定的统一编译命令生成 bridge 动态库和 seekdb。可用 [DDL 持久化验收脚本](validation/run_ddl_persistence_smoke.sh) 检查离线建索引、首次检索和重启后的 GPU 查询，并用 [构建测量脚本](validation/measure_offline_build.py) 在同一台 Radeon 上对照冻结 V5。

[赛题与验收规则](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/seebnbaoxot1b0tb)

## 验证范围

最终图邻接度为 128、中间图邻接度为 256，NN-Descent 迭代上限为 32。Top-K 大于 64 时 itopk_size 下限为 768，其余查询为 512，且至少为 Top-K 的四倍。search_width 为 4。参数调整的实际召回和速度须由 Radeon 实测确认。

主机侧工作线程测试覆盖 32,000 次多线程提交、返回值、线程复用及重复创建和销毁，并通过 ASan/UBSan。固定版本头文件下的 bridge 和启用 CUVS 的适配器对象编译通过。完整 CPU 构建和 VSAG SQL 回归可在本地执行。Radeon 上的 Recall、GPU 运行与性能需要实机或正式评测确认。

主机侧 bridge 测试覆盖 Top-K 切换、重复查询、缓冲区扩容、大批量临时内存、四路并发、分配和搜索及复制失败恢复，以及索引重载后的释放。模拟 GPU API 仅验证资源生命周期与参数传递，召回验证需要真实 hipVS 执行。
