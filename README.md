# LightVLN — 修复前代码快照

[项目主页 / Project website](https://lightvln-aerial-navigation.warmgame4.chatgpt.site)

版本：`pre-fix-20261002`。这是 LightVLN Local Weighted32 + Native1T H16 + history-conditioned instruction-selection residual 的私有归档，用于后续修复前后的可追踪比较。

保留模型、数据读取、训练与评估基础代码及 AirVLN/OpenFly 配置。**不包含模型权重、训练数据、轨迹、缓存、日志、视频或第三方源码。** `aerial_openvla` 是历史包名。

- [训练与评估说明](docs/TRAINING.md)
- [第三方依赖](docs/THIRD_PARTY.md)
- [来源与阶段配置](docs/PROVENANCE.md)
- [修复前已知问题与复现边界](docs/KNOWN_ISSUES.md)
- [源码 SHA-256 清单](docs/source_manifest.json)

## 安装与检查

使用 Python 3.10+、Linux/CUDA 训练环境；先按第三方说明配置原实验依赖。

```bash
python -m pip install -e . --no-deps
python tools/repo_guard.py
python tools/validate_snapshot.py
```

配置中的 `/external/lightvln` 是待替换路径，不会自动下载数据。可以生成机器专用配置：

```bash
python tools/materialize_configs.py --root /absolute/path/to/external/lightvln --output local_configs
```

`local_configs/` 被 Git 忽略。数据和权重应放在仓库外。新仓库仅做静态源码/配置验证，未在此快照重新运行 GPU 训练或闭环评估；不能把归档当作已重新验证的实验结果。

提交前可启用文件检查：`git config core.hooksPath .githooks`。检查拒绝非白名单路径、大文件、二进制和常见凭据格式；它不能代替人工审核。
