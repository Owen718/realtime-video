# 视频到视频（V2V）与降噪日程

- 文件：`v2v.py`
- 功能：
  - 读视频（支持 URL），自动 16fps 重采样与旋转元数据处理
  - 编码为潜空间（与 VAE 编码器配合）
  - 动态生成 `denoising_step_list` 以匹配 `strength`

## 加载与预处理

- `load_video_as_rgb(path_or_url, resample_to=16)`：
  - 使用 `ffprobe` 读取旋转信息，必要时旋转帧
  - 过长视频按帧数阈值自动用 ffmpeg 重采样到 16fps
  - 返回 `[T, 3, H, W]` 的 FloatTensor，范围 `[-1,1]`

## 潜空间编码

- `encode_video_latent(vae, encode_vae_cache, resample_to, max_frames, frames, height, width, stream)`：
  - 对齐分辨率到 VAE stride（8 的倍数）
  - 双缓冲式逐段编码，返回 `[T, 16, H/8, W/8]` 与编码缓存
  - 服务侧用于：
    - `input_video`：整体编码后与噪声按 `strength` 线性混合初始化
    - `webcam_mode`：9/12 帧批次流式编码，保持与 3 帧生成块对齐

## 动态降噪步长

- `get_denoising_schedule(timesteps, denoising_strength, steps)`：
  - 在 `[denoising_strength*1000, 0]` 间线性取 `steps` 个点
  - 使用训练 `timesteps` 做反向映射，得到推理时的整数步长列表

