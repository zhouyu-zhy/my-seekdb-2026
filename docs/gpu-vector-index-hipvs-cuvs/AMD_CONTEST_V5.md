# AMD 赛道 V5

本版本沿用官方 V5 的 FP32/L2 CAGRA 构建参数、OBCUVS02 快照和原生 hipVS 序列化与重载接口。快照的临时文件复制按 1 MiB 分块，降低大图持久化时的读写调用次数。批量搜索一次申请设备临时内存，为查询向量、ID 和距离分别保留 256 字节对齐的地址。距离复制失败时返回错误。

使用比赛规定的统一编译命令生成 bridge 动态库和 seekdb。可用 [DDL 持久化验收脚本](validation/run_ddl_persistence_smoke.sh) 检查离线建索引、首次检索和重启后的 GPU 查询，并用 [构建测量脚本](validation/measure_offline_build.py) 在同一台 Radeon 上对照冻结 V5。

[赛题与验收规则](https://oceanbase.yuque.com/org-wiki-obtech-vh7w9r/hc3pio/seebnbaoxot1b0tb)
