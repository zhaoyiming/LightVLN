# 训练与评估入口

以下命令在仓库根目录执行。先安装原环境依赖与本包，生成 `local_configs`，并逐项核对数据、初始化 checkpoint、Stage1 选择点和 cache 路径。原始 manifest 内的图片路径也必须有效；路径替换工具不会修改外部数据。

## AirVLN

```bash
python tools/materialize_configs.py --root /absolute/path/to/external/lightvln --output local_configs
python -m torch.distributed.run --standalone --nproc_per_node=2 -m aerial_openvla.training.train --config local_configs/airvln/base.yaml
python -m torch.distributed.run --standalone --nproc_per_node=2 -m aerial_openvla.training.train --config local_configs/airvln/stage1.yaml
python scripts/cache_native1t_postfusion.py --config local_configs/airvln/stage1.yaml --checkpoint /path/to/selected-stage1 --output /path/to/cache
python -m torch.distributed.run --standalone --nproc_per_node=2 -m aerial_openvla.training.train --config local_configs/airvln/stage2.yaml
```

这些是依次执行的阶段入口，不能并发启动整条链。Stage2 的 `init_checkpoint` 来自 Base，memory tokenizer 来自选定 Stage1，cache 必须与该 tokenizer 对应。历史 Stage1 选择 step 500，而非默认取最终 epoch；需按验证证据选取，不能直接把最终 checkpoint 改名。

Base/Stage2 单卡 batch 64 × 2 GPU × 梯度累积 2 = global batch 256；Stage1 单卡 32 × 2 × 4 = 256。Stage1 冻结策略主体，训练 Native1T tokenizer/projector/MEM_BOS；Stage2 冻结 tokenizer，训练 LoRA、pooler/HRIS、投影与动作头。准确冻结逻辑保留在模型源码。

## OpenFly

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_openfly_minivla_observed8.py --config local_configs/openfly/base.yaml
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_openfly_instruction_pooler32_native1t_stage1_tfds.py --config local_configs/openfly/stage1.yaml --openfly-tfds-dir /path/to/tfds
python scripts/cache_openfly_pooler32_native1t_tfds_parallel.py --config local_configs/openfly/stage1.yaml --checkpoint /path/to/selected-stage1 --parent-checkpoint /path/to/base-epoch3 --tfds-dir /path/to/tfds --output /path/to/cache
python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/train_openfly_instruction_pooler32_native1t_stage2_tfds.py --config local_configs/openfly/stage2.yaml --openfly-tfds-dir /path/to/tfds
```

这些入口有严格历史 contract 检查：完整训练集 1,608,400 frames / 100,221 episodes；Stage1 选择记录和缓存校验材料须来自匹配实验。详见脚本中的 `require_selected_stage1` 与 provenance 检查。本版本保留校验，不包括生成所有历史 gate 材料的集群调度工作流。

OpenFly compact 8 对应原始动作 ID `(0,1,2,3,4,5,8,9)`；评估应还原官方 ID，不能按 AirVLN 的 8 类动作解释。

## AirVLN 评估

```bash
AIRVLN_AUDIT_DIR=/path/to/eval-audit AIRVLN_WORKER_ID=0 python scripts/eval_airvln_verified.py --config local_configs/airvln/eval_unseen.yaml
```

先核对 checkpoint、仿真服务、外部 AirVLN/OpenVLA 根目录。入口保留原 LoRA 检查，仅支持历史 step 13563/18084。当前 YAML 是单 worker 的 unseen 配置，不是全量多卡分片启动器。OpenFly 评估需在外部官方平台挂接 `aerial_openvla.openfly.policy_adapter`，不在此快照重复收录平台。
