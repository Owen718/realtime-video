# 视频 VAE 与流式缓存

- 解码封装：`demo_utils/vae_block3.VAEDecoderWrapper`
- 编码封装（服务侧可选）：`demo_utils/vae_block3.VAEEncoderWrapper` 与 `wan/modules/vae._video_vae`
- 原始 Wan VAE：`wan/modules/vae.py`（推理默认仅解码端使用）

## 设计目标

- 在实时场景中以最小延迟将 `[B, T, 16, H/8, W/8]` 解码到 `[B, T, 3, H, W]`。
- 借助分块与特征缓存，避免跨块重复计算（上采样与时间卷积层明显受益）。

## 接口与缓存

- `VAEDecoderWrapper.__call__(z, *feat_cache)`：
  - 输入 `z`：`[B, T, 16, H/8, W/8]`，内部转置为 `[B, 16, T, H/8, W/8]`。
  - `feat_cache`：长度 55 的多级特征缓存列表；返回更新后的缓存以复用到下一个块。
  - 输出：像素帧与新缓存：`(pixels, feat_cache)`。
- `VAEEncoderWrapper(z, feat_cache, stream)`：视频编码为潜空间，支持流式编码（Webcam/V2V）。

### 特征缓存的层级与访问顺序

- 缓存数组长度约 55，对应编码/解码网络中若干 `CausalConv3d`、Resample、ResidualBlock 等层的时间状态（`demo_utils/vae_block3.py`）。
- 访问顺序由模块前向时的 `feat_idx[0]` 自增驱动，保证每一层在相同位置读/写缓存：
  - 编码端：`VAEEncoderWrapper.forward(...)` 中首段为单帧，后续每次追加 4 帧（与 3 帧生成块对齐，首段用于引导因果卷积，`demo_utils/vae_block3.py:56-94`）。
  - 解码端：逐帧迭代，将每帧的中间状态写入对应缓存槽位，下一块会复用（`demo_utils/vae_block3.py:199-239` 及 `256-309`）。

### CausalConv3d 与 Resample 的流式实现

- `CausalConv3d`：时间维只看过去（含当前）的窗口；配合缓存实现跨块“延续”（`demo_utils/vae_block3.py:146-189`, `306-339`）。
- `Resample`：
  - `upsample3d` 模式下，会使用时间卷积将 T 维扩展为两倍，并更新缓存（`Resample.forward` 内部的 `time_conv` 分支）。
  - `downsample3d` 模式下，将当前块最后 1 帧缓存起来，供下一块拼接做因果下采样。

## 归一化与尺度

- 使用注册的 `mean/std` 在潜空间与像素空间之间做线性变换（与 Wan VAE 对齐）。
- 解码输出范围为 `[-1, 1]`，服务侧在发送前转为 `[0,1]`。

## 与推理管线的配合

- 推理每块结束后立刻解码并推送帧，`release_server.py` 在默认分辨率 `(832,480)` 时启用编译优化，其余分辨率回退 eager。
- `kv_cache_num_frames` 限制了用于 KV 刷新的“干净上下文”帧数，VAE 不直接受此参数影响，但更短上下文可降低端到端时延与显存占用。

### 编译与形状限制

- 服务端默认在 `(832,480)` 时 `torch.compile` VAE 解码器，以充分受益内核融合；
- 其他分辨率将强制 `force_eager`，避免动态图导致的图分裂与编译回退（`release_server.py:712-719`）。
