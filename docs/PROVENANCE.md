# 来源与版本

整理日期：2026-10-02；模型/训练/评估源码来自 2026-09-09 至 2026-09-12 留存的实验快照，而非整理当天服务器 HEAD。整理时 HPC3 SSH 拒绝连接，使用已有本地证据。

核心 `src/aerial_openvla` 来自 AirVLN 实验 611352 的冻结评估源码。AirVLN cache 入口来自该训练链参考源码。OpenFly 支持入口来自 2026-09-09 训练快照。`source_manifest.json` 记录每个归档 Python 文件的来源标签、大小与 SHA-256；它用于完整性核对，并非数字签名。

模型和评估算法源码逐字节保留。新增内容仅包括包装工具、项目元数据与文档；YAML 中机器绝对路径改为外部路径占位符。OpenFly Stage2 同时还原 607858 启动器实际覆盖的 HRIS 开关、4 epochs、4-epoch scheduler horizon、无 max_train_steps。未带入硬编码 sbatch 自动链。

| 训练链 | Base | Stage1 | Stage2 |
|---|---|---|---|
| AirVLN 611352 | 3 epochs，LR horizon 5；E3 step 13563 | 最多 1 epoch，历史实际停止于 1020，选 step 500 | 4 epochs；每 epoch 4521 updates；E3 step 13563 |
| OpenFly 607858 | 3 epochs；E3 step 18849 | 1 epoch 6283 updates，历史选 step 6000 | 4 epochs；每 epoch 6283 updates；E3 step 18849 |

AirVLN 与 OpenFly 是分别训练的权重，动作语义也不同。历史选择点不是本仓库重新得出的结论。AirVLN stage1 YAML 保留原始计划，不包含外部人工/脚本提前停止操作。

架构为当前帧 256→32 tokens、4×8 空间分组；历史最多 16 帧，每帧压缩为 1 token。历史 tokenizer 使用 8-head MHA。HRIS temporal MLP 为 16→32→1；γ 为初始化为零的无约束参数。完整实现以模型源码为准。

本仓库不是整个工作目录备份；CMA/Qwen35 等非 LightVLN 分支可能仍有历史接口但其完整模型实现不在此版本范围内。
