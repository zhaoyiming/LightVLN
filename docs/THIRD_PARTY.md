# 第三方依赖与外部资产

此仓库不 vendor 第三方项目、不保存权重；所有上游的许可证与权重/数据使用条款仍适用。本私有快照不新增公共开源授权。

| 外部项 | 用途 / 获取线索 |
|---|---|
| PyTorch / torchvision | CUDA 训练与图像模型；版本须与原环境一致 |
| Transformers / PEFT / Accelerate | Hugging Face 模型、LoRA 与设备支持 |
| timm / Pillow / NumPy / einops | 视觉 backbone、图像及张量处理 |
| PyYAML / tqdm / h5py / pyarrow | 配置、进度、缓存与 parquet 读取 |
| FlashAttention 2 | 训练配置启用；必须与 CUDA/PyTorch 匹配 |
| Prismatic / OpenVLA | 自定义多模态 backbone，代码中的 `prismatic` 导入与 HF remote code；须使用实验匹配的 checkout |
| OpenFly-Platform | 官方训练数据/模拟器与闭环协议；https://github.com/SHAILAB-IPEC/OpenFly-Platform |
| AirVLN + AirSim/UE | AirVLN 数据、仿真环境与外部 Python 客户端；按原实验环境准备，源码不会随本仓库上传 |
| Prism-Qwen2.5-0.5B HF 资产 | tokenizer、processor、配置、自定义模型代码与初始化权重，放在仓库外 |

`pyproject.toml` 的 extras 只是根据源码导入整理的依赖清单，**不是原实验的版本锁文件**。没有完整可核实的 `pip freeze` 和全部第三方 commit，故未虚构版本号或精确复现保证。应先在原训练环境导出版本清单并核对自定义模型，再运行训练。

OpenFly TFDS 读取器含项目自己的解析实现；数据、索引、cache provenance、checkpoint 选择元数据仍须外部准备。不得用空文件绕过检查。
