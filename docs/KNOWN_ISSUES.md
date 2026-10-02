# 修复前已知问题

本版本刻意保留历史行为，后续修复应提交独立 commit，不覆盖这个快照。

1. AirVLN 配置包含 `reset_discard_frames: 60`，但历史 rollout/adapter 没有把该值真正传入 reset 图像处理链。配置值不能证明已丢弃 60 帧；warmup 为 0。
2. 已检查的 HRIS checkpoint γ 约为 -0.00318、选择注意力接近均匀。该观测并不能单独证明 unseen 失败的因果关系。
3. 当前局部池化的指令键来自词嵌入，池化阶段对词序不敏感；这不代表后续语言模型不处理词序。
4. 历史训练配置 `val_manifest: null`，没有训练期 held-out 验证。
5. 评估需验证 LoRA 确实生效。`scripts/eval_airvln_verified.py` 保留原审计入口的 adapter effect、safe merge、输出近似一致及 token-shape 检查；限定历史 E3/E4 step，不是通用 checkpoint 加载器。
6. OpenFly 支持脚本包含固定数据量、父 checkpoint、selected_stage1 和 cache provenance 约束。归档使用冻结 AirVLN 核心搭配历史 OpenFly 支持脚本；组合尚未重新跑 GPU 验证。缺少匹配元数据时应停止，不能删掉 assert 伪造兼容性。
7. 基础 Prism/Qwen2.5-0.5B 自定义 Hugging Face 模型、processor、trust_remote_code 文件与匹配训练环境必须另行准备。普通上游同名模型不保证等价。

静态语法和配置检查不等于 checkpoint 数值等价或评估结果复现。本版本不宣称修复 unseen 表现。
