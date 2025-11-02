# 配置项与环境变量说明

- 默认配置：`configs/default_config.yaml`
- 14B 服务配置：`configs/self_forcing_server_14b.yaml`
- 环境变量：`settings.py`、`release_server.py` 读取

## 关键配置项

- `denoising_step_list`：推理时间步列表（训练时刻 0-1000 对应 FlowMatch 映射）；服务侧可动态重采样（见 `v2v.get_denoising_schedule`）。
- `warp_denoising_step`：将 `denoising_step_list` 由训练坐标系 warp 到调度器坐标系（需要去掉末尾 0）。
- `timestep_shift`：FlowMatch sigma-shift，影响速度/质量权衡（大一些可更快）。
- `num_frame_per_block`：每块生成的帧数，通常为 3。
- `independent_first_frame`：是否将首帧独立成单帧块（I2V 场景有用）。
- `context_noise`：KV 刷新时给“干净上下文”注入的小噪声，稳定性/一致性平衡。
- `local_attn_size`（通过 `model_kwargs` 传入模型）：局部注意力窗口，限制可见历史长度，降低显存与延迟。
- `checkpoint_path`：自回归权重（`krea-realtime-video-14b.safetensors`）。
- `enable_fp8`：通过 `torchao` 对 Transformer 动态激活/权重量化为 FP8（显存/吞吐换质量）。

## 环境变量

- `MODEL_FOLDER`：Wan 模型根目录（含 `Wan2.1-T2V-1.3B/` 与权重文件）。
- `CONFIG`：加载的 YAML 配置路径。
- `DO_COMPILE`：`true/false`，启用 `torch.compile`；分辨率与形状需匹配 `settings.COMPILE_SHAPES` 才能充分收益。
- `USE_STATIC_ENCODER_COND_DICT`：调试用途，返回固定文本条件（请勿用于生产）。

