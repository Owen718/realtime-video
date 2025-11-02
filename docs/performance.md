# 性能优化与显存策略

## 目标平台与指标

- B200（推荐）：4 步推理约 11 fps（832x480）
- H100 / RTX 5xxx：支持，注意后端选择与编译策略

## 关键旋钮

- 分辨率：`width/height` 需为 8 的倍数；分辨率越高，帧内 token 越多（`frame_seq_length=1560` 对应 832x480）。
- 步数：`num_denoising_steps` 越少越快（如 4）；质量受影响，可结合 `strength` 与提示工程权衡。
- 块大小：`num_frame_per_block=3` 为实时/延迟的折中；过小会频繁刷新 KV，过大延迟提升。
- KV 窗口：`kv_cache_num_frames`/`local_attn_size` 限制回看的帧数，缩短 KV；长视频生成时尤为重要。
- 后端：B200 推荐 FlashAttention 4；H100/RTX 5xxx 可选 SageAttention（见 attention-backends.md）。
- 编译：`DO_COMPILE=true` 结合固定分辨率可提升吞吐；非固定分辨率回退 eager。
- 量化：`enable_fp8` 可进一步降低显存并提升吞吐，质量略降。

## 显存粗估

- KV Cache：约 `num_blocks * heads * head_dim * kv_len`，14B 下单卡可达 20~25GB。
- 文本跨注意力缓存：固定 512 token，影响较小。
- VAE：解码端缓存 50+ 层特征，按块复用，显存稳态占用有限。

## 实践建议

- 生产环境固定分辨率与步数，预编译图；只暴露必要参数给前端。
- 长视频生成：启用局部注意力/限制 KV 帧数；按需滑窗刷新 `cache_start`。
- WebSocket 帧压缩：JPEG 质量在 80~90 之间可兼顾带宽与观感。
- I/O 并行：独立 `torch.cuda.Stream` 上传/下载，避免与推理主流抢占。

