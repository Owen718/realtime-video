# 训练核心总览（Self‑Forcing / DMD / SiD / GAN / ODE）

本篇系统整理仓库内「已实现的训练核心」：包含自回归化训练（Self‑Forcing）的一致性回放管线、因果主干与缓存、调度与损失、以及四类训练目标（DMD / SiD / GAN / ODE 回归）的接口与使用方式。可直接作为你拼装训练循环的参考蓝图。

- 面向读者：算法与工程同学，计划基于本仓库进行训练/蒸馏/对齐。
- 关联阅读：
  - 推理与管线：docs/inference-causal.md
  - 模型组件：docs/components-model.md
  - Self‑Forcing 概述：docs/training-self-forcing.md
  - 性能与注意力后端：docs/attention-backends.md, docs/performance.md

## 总体设计与术语

- Self‑Forcing 训练：在训练端用“推理轨迹”展开模型（Consistency Backward Simulation），通过 KV/CrossAttn 缓存 + 块级因果掩码，逐块生成并在非反传步上重噪（re-noise），最大化训练/推理一致性。
- 块级生成：`num_frame_per_block`，常取 3。每块内部按 `denoising_step_list` 循环时间步；仅最后一步留作真实预测，其余步用调度器把 x0 预测重投影到下一步。
- KV/CrossAttn 缓存：每个 Transformer Block 独立维护 `k/v` 和文本跨注意力缓存；块结束后再用“干净上下文”刷新一次缓存以对齐下一块的可见历史。
- FlowMatch 视角：统一使用 `FlowMatchScheduler` 将 `sigma∈[sigma_min, sigma_max]` 映射到训练时间步；通过封装的 x0<->flow 转换在不同目标之间切换。

## 关键模块一览

- Self‑Forcing 一致性回放（训练侧）
  - pipeline/self_forcing_training.py: `SelfForcingTrainingPipeline`
    - 初始化缓存：`_initialize_kv_cache`/`_initialize_crossattn_cache`。
    - `inference_with_trajectory(...)`：
      - 支持 `independent_first_frame`、`same_step_across_blocks`、`last_step_only`、`context_noise` 等策略；
      - 非目标步 `no_grad`，目标步反传；
      - 块结束后以“干净上下文”重跑刷新缓存（可裁剪上下文长度）。

- 训练入口与公共封装
  - model/base.py: `BaseModel` / `SelfForcingModel`
    - 统一装配：`WanDiffusionWrapper`（生成器/打分器）、`WanTextEncoder`、`WanVAEWrapper`、`FlowMatchScheduler`；
    - `SelfForcingModel._run_generator(...)` 内部调用 `SelfForcingTrainingPipeline` 展开生成器，返回预测、梯度 mask、回放时间区间（供时刻调度）。

- 训练目标（四类）
  - model/dmd.py: `DMD`
    - 分布匹配蒸馏：利用非因果老师（`real_score`，可 CFG）与因果学生（`fake_score`）在同一 noisy 输入上的 x0 预测差构造 KL‑grad，再以 MSE 对“生成样本”做一阶近似更新。
  - model/sid.py: `SiD`
    - 自诱导蒸馏：直接在生成样本的 noisy 版本上计算老师/学生差异，轻量对齐（不额外展开生成器）。
  - model/gan.py: `GAN`
    - 在 noisy 潜空间上训练分类分支充当判别器；支持相对/非相对判别器 + R1/R2 正则；生成器端以对抗项优化。
  - model/ode_regression.py: `ODERegression`
    - 从预生成的 ODE 轨迹随机抽步回归到目标 x0；无需真实数据集即可训练。

- 模型与封装
  - utils/wan_wrapper.py: `WanDiffusionWrapper` / `WanTextEncoder` / `WanVAEWrapper`
    - 统一加载因果/非因果 Wan 主干；FlowMatch 调度器；`flow_pred <-> x0_pred` 双向转换；文本编码与视频 VAE 封装。
  - wan/modules/causal_model.py: `CausalWanModel`
    - 块掩码（Flex Attention `BlockMask`）、KV/CrossAttn 缓存、3D RoPE、可选局部注意力窗口 `local_attn_size`、QKV 融合。

- 调度与损失
  - utils/scheduler.py: `FlowMatchScheduler`
    - `set_timesteps`、`add_noise`、`step`、训练加权；以及 x0/v/noise 的互转接口（由 `SchedulerInterface` 注入）。
  - utils/loss.py: 去噪损失工厂
    - `x0`、`v`、`noise`、`flow` 四类（FlowMatching 权重已内置）。

- 数据与工具
  - utils/dataset.py: LMDB 分片读取（Sharding/ODERegression）、简单文本数据集与图文对；
  - scripts/generate_ode_pairs.py: 生成/存储 ODE 采样轨迹的示例脚本；
  - utils/distributed.py: FSDP 封装、全量 state_dict 导出、EMA 工具。

## 形状与常量（速查）

- 潜空间张量：`[B, T, C=16, H/8, W/8]`
- 每帧 token 数：`frame_seq_length = 1560`（例如 480×832 → 60×104，Patch stride=(1,2,2) → 30×52）。
- 块长度：`num_frame_per_block`（默认 3）。
- 时间步列表：`denoising_step_list`（训练刻 0–1000 映射到调度器刻，通过 `warp_denoising_step` 可扭曲对齐）。

## 训练循环如何拼装（最小配方）

以下示例展示如何用现有接口搭起一个 DMD 的最小训练循环（伪代码，省略日志/评估/断点）。

```python
import torch
from easydict import EasyDict as edict
from model.dmd import DMD

# 1) 准备超参（可参考 configs/default_config.yaml）
args = edict(
    num_train_timestep=1000,
    denoising_step_list=[1000, 937, 833, 625, 0],
    warp_denoising_step=False,
    timestep_shift=5.0,
    num_frame_per_block=3,
    independent_first_frame=False,
    same_step_across_blocks=True,
    last_step_only=False,
    num_training_frames=21,
    guidance_scale=3.0,
    denoising_loss_type='flow',
    gradient_checkpointing=True,
    mixed_precision=True,
    context_noise=0,
    model_kwargs=dict(timestep_shift=5.0),
)

# 2) 构造模型（含生成器/老师/学生打分器、文本编码器与 VAE 封装）
device = torch.device('cuda', 0)
model = DMD(args, device).to(device)

# 3) 优化器（示例：仅训生成器与 fake_score）
opt = torch.optim.AdamW([
    {'params': model.generator.model.parameters(), 'lr': 1e-4},
    {'params': model.fake_score.model.parameters(), 'lr': 1e-4},
], weight_decay=0.01)

# 4) 训练步
model.train()
for step in range(steps):
    # 文本条件（正向/负向，负向可自定义）
    cond = model.text_encoder(["a cinematic shot of ..."])  # dict: {'prompt_embeds': ...}
    uncond = model.text_encoder(["负向提示词 ..."])        

    # 指定输出潜空间形状（真实数据不必提供）
    B, T, H, W = 1, 21, 480, 832
    image_or_video_shape = [B, T, 16, H//8, W//8]

    # 生成器展开 + DMD 损失（内部做一致性回放与时间步调度）
    loss_g, log = model.generator_loss(
        image_or_video_shape=image_or_video_shape,
        conditional_dict=cond,
        unconditional_dict=uncond,
        clean_latent=None,
        initial_latent=None,   # I2V/视频扩展时可传入参考潜空间
    )

    opt.zero_grad(set_to_none=True)
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

- 判别器/打分器：如需对抗分支或“假分数器”训练，分别调用 `GAN.critic_loss(...)` 或 `DMD.critic_loss(...)` 按常规 GAN 方式交替更新。
- ODE 回归：用 `ODERegression.generator_loss(ode_latent, cond)`，其中 `ode_latent` 可用 `scripts/generate_ode_pairs.py` 预生成并通过 `utils/dataset.ODERegressionLMDBDataset` 读取。

## 关键超参与建议

- 时间步与调度：
  - `denoising_step_list` 建议覆盖头/中/尾若干关键时间步；服务侧默认 `[1000,937,833,625,0]`。
  - `timestep_shift>1` 可加快收敛/推理速度；注意与训练刻度匹配，必要时打开 `warp_denoising_step`。
- 块与上下文：
  - `num_frame_per_block=3` 是延迟与质量的折中；I2V 可启用 `independent_first_frame`。
  - `context_noise` 用于“干净上下文”刷新时的小噪声注入，稳定性/一致性权衡。
- 显存策略：
  - 主要消耗在 KV 缓存与注意力；通过 `local_attn_size` 限制历史窗口长度可显著降显存（在 `model_kwargs` 传入因果模型）。
  - 开启 `gradient_checkpointing` 与 QKV 融合（`fuse_projections`）提升可训练规模；结合 `torch.compile`/FP8 见 docs/attention-backends.md。

## 训练目标要点

- DMD（model/dmd.py）
  - 学生 `fake_score` 与老师 `real_score` 共享结构，但老师固定非因果、不开梯度；支持 CFG。
  - `ts_schedule`/`ts_schedule_max` 可在生成器回放时间区间内截断采样，减少时刻错配。
- SiD（model/sid.py）
  - 轻量自蒸馏，不额外展开生成器，适合资源更紧的场景。
- GAN（model/gan.py）
  - 在 `fake_score` 上动态添加分类分支（`adding_cls_branch`），对 noisy 潜空间分类；支持 R1/R2 正则与相对判别器变体。
- ODE 回归（model/ode_regression.py / scripts/generate_ode_pairs.py）
  - 从 ODE 采样轨迹 `ode_latent[:, -1]` 回归目标 x0；可脱离真实数据训练自回归器或热启动。

## 「已有 vs. 需自备」

- 已有：
  - 一致性回放管线、因果主干、调度与损失、四类训练目标、FSDP/EMA、LMDB 数据读取与 ODE 轨迹脚本。
- 需自备：
  - 外层训练脚手架（分布式启动、优化器/调度器、日志、断点）；
  - 数据（若采用 ODE 回归或有监督对齐）；
  - 评估指标与可视化。

## 故障排查（FAQ）

- 显存爆：
  - 降低分辨率/批量；设置较小 `local_attn_size`；减少 `num_frame_per_block`；打开混精与梯度检查点。
- 生成漂移/不稳定：
  - 调小 `context_noise`；缩短“干净上下文”长度（`SelfForcingTrainingPipeline` 中的 `max_num_context_frames`/`keep_first_frame`）。
- 老师/学生时刻错配：
  - 结合回放区间开启 `ts_schedule`/`ts_schedule_max`，或启用 `warp_denoising_step` 进行坐标系扭曲对齐。

---

如需，我们可以在此基础上提供一个可运行的最小 `train.py`（含 FSDP/EMA/断点与日志），并给出若干配置范式与多机多卡启动脚本示例。

