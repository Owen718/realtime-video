# FAQ / 故障排查

## 服务无法启动 / 权重加载失败

- 确认 `MODEL_FOLDER` 指向包含 `Wan2.1-T2V-1.3B/` 的目录。
- 确保已下载：
  - `Wan-AI/Wan2.1-T2V-1.3B`（子目录包含 T5/Tokenizer/VAE 权重）
  - `krea/krea-realtime-video` 中的 `krea-realtime-video-14b.safetensors` 到 `checkpoints/`
- CUDA 驱动与 `torch` 版本需匹配。

## 显存不足 / 性能不达标

- 降低分辨率或步数 `num_denoising_steps`。
- 限制 KV 窗口：
  - 设置 `local_attn_size`（通过 `model_kwargs`）
  - 缩短 `kv_cache_num_frames`（服务端会截断“干净上下文”长度）
- 使用合适后端：B200→FlashAttention 4；其他→SageAttention。
- 启用 `DO_COMPILE=true` 并固定形状；必要时关闭以提升稳定性。
- 可尝试 `enable_fp8=true`（质量略降）。

## 视频输入/输出问题

- 上传的视频会被重采样到 16fps（过长时），注意节奏变化。
- 下载 MP4 使用 `ffmpeg` 构建，确保环境中可用。

## 质量问题

- 提示词工程：更明确的主体/动作/镜头/风格描述通常更好。
- 提高步数与 `guidance_scale`，或改用离线 `sample.py` 批量生成并做主观筛选。
- 若使用 SageAttention 2++/3，可能质量下降，建议回退 2.2.1。

