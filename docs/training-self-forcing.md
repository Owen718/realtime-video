# Self-Forcing 训练与一致性回放

- 训练相关：`model/base.py`, `model/causvid.py`, `pipeline/self_forcing_training.py`
- 思想：将扩散网络蒸馏为自回归生成器，通过缓存与一致性回放减小训练/推理分布差异

## 关键组件

- `SelfForcingTrainingPipeline`：
  - 作用：在训练时模拟推理轨迹（Consistency Backward Simulation），避免 teacher 与 inference 分布偏移。
  - 步骤：
    1) 初始化 KV/CrossAttn 缓存；
    2) 若有 `initial_latent` 先以 `t=0` 写入缓存；
    3) 逐块逐步运行，除最后一步外使用 `add_noise` 回退到下一步；
    4) 得到输出后，以“干净上下文”（可注入 `context_noise`）重跑刷新缓存；
    5) 返回轨迹（或部分）与对应的去噪步编号，用于一致性/匹配损失。
- `CausVid`（`model/causvid.py`）：
  - 支持 DMD（Distribution Matching Distillation）梯度，计算 `fake_score` 与 `real_score` 间差异；
  - 在 `num_frame_per_block>1` 时与块掩码配合，保持分块自回归语义。

## 损失与时间采样

- `denoising_step_list`：在训练时也用于从多个时间步采样输入；
- 支持 FlowMatching / DMD 等多种目标（见 `utils/loss.py`）。

## 训练/推理一致性

- 训练端使用与推理端一致的块掩码与缓存写入策略；
- 通过“重噪（re-noise）→再去噪”的回放，最小化与推理端推演路径差异。

