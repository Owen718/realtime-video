# 整体架构与数据流

本项目实现了基于 Self-Forcing 的 14B 级实时视频扩散推理。核心由三层组成：

- 文本编码（UMT5-XXL）：将自然语言转换为上下文特征，供跨注意力使用。
- 因果扩散主干（CausalWanModel）：块级因果注意力、KV Cache 与跨注意力缓存，逐块生成视频潜空间帧。
- 视频 VAE：将潜空间 `[B, T, C=16, H/8, W/8]` 解码为像素帧并流式输出。

## 端到端流程（实时服务）

1) 加载阶段
- `release_server.py` 在应用生命周期 `lifespan` 中读取 `configs/self_forcing_server_14b.yaml` 并加载：
  - 文本编码器 `utils/wan_wrapper.WanTextEncoder`
  - 扩散主干 `utils/wan_wrapper.WanDiffusionWrapper`（内部使用 `wan/modules/causal_model.CausalWanModel`）
  - VAE 封装 `demo_utils/vae_block3.VAEDecoderWrapper`（解码端为流式缓存）
  - 因果推理管线 `pipeline.CausalInferencePipeline`
- 可选：`DO_COMPILE=true` 时对 Transformer/VAE 进行 `torch.compile`。

2) 会话与参数
- WebSocket 端点 `/session/{id}` 接收 Prompt 与参数，创建 `GenerationSession`：
  - 分辨率对齐到 8 的倍数（VAE stride），计算潜空间尺寸。
  - 初始化 `all_latents` 与 `noise`，形状 `[1, num_blocks*3, 16, H/8, W/8]`。
  - 根据 `strength` 与步数 `num_denoising_steps` 生成动态的 `denoising_step_list`（见 `v2v.get_denoising_schedule`）。

3) 块级生成（Self-Forcing 推理）
- 以 `num_frame_per_block=3` 为单位迭代：
  - 对当前块，按 `denoising_step_list` 从大到小时间步循环：
    - 输入 `noisy_input`、`timestep`、`conditional_dict`（含 `prompt_embeds`）、`kv_cache`、`crossattn_cache` 给 `WanDiffusionWrapper`。
    - 非最后一步：用 `FlowMatchScheduler.add_noise` 将输出 `pred_x0` 重新扰动到下一时间步（“re-noise”）。
    - 最后一步：得到该块最终潜空间预测 `denoised_pred`。
  - 更新全局潜空间 `all_latents` 与最近预测缓存。
  - 使用 `VAEDecoderWrapper` 流式解码该块像素帧，并通过回调发送到客户端。
  - 使用“干净上下文”再次以 `context_noise` 近零时间步运行一遍，刷新 KV Cache，使下一块能引用当前块内容。

4) KV Cache 与跨注意力缓存
- `pipeline/causal_inference.py` 初始化：
  - 每个 Transformer Block 分配一组 KV 缓存字典 `{k, v, global_end_index, local_end_index}`。
  - 每个 Block 同时分配跨注意力缓存 `{k, v, is_init}`（用于文本上下文）。
- KV 长度由 `frame_seq_length=1560` 与窗口策略决定：
  - 全局注意力：`kv_cache_size ≈ num_frames * 1560`
  - 局部注意力：`kv_cache_size ≈ local_attn_size * 1560`
- 推理时每步将当前帧的 K/V 写入缓存，并依据块掩码只允许关注到当前块结尾之前的 token。

5) 流式 I/O 与会话管理
- GPU->CPU 异步复制并 JPEG 编码，逐帧/逐块通过 WebSocket 推送给前端。
- 支持上传视频/首帧、下载最终 MP4、以及 Webcam 输入模式。

## 模块边界与数据形状

- 文本编码器：
  - 输入：`List[str]` Prompt
  - 输出：`{"prompt_embeds": Tensor[L, C]}`，长度 512，维度 4096（UMT5-XXL）
- 扩散主干：
  - 输入：`noisy_image_or_video` `[B, T, 16, H/8, W/8]` 与 `timestep` `[B, T]`
  - 输出：`(flow_pred, pred_x0)`，其中 `pred_x0` 为 x0 预测（通过 FlowMatch 公式转换）
- VAE 解码：
  - 输入：`[B, T, 16, H/8, W/8]`
  - 输出：像素 `[B, T, 3, H, W]`，范围经 `(x*0.5+0.5)` 归一化到 `[0,1]`

## 关键设计点

- 块级因果注意力：通过 `wan/modules/causal_model.get_block_mask` 构建按块右上三角可见性的掩码，满足自回归视频生成。
- FlowMatch 调度：`utils/scheduler.FlowMatchScheduler` 将 sigma 映射到训练时间步，支持额外终点、反向/反转等变体。
- VAE 流式缓存：`demo_utils/vae_block3.VAEDecoderWrapper` 维护多级特征缓存，减少跨块重复计算，显著降低解码延迟。
- 灵活注意力后端：默认 PyTorch Flex Attention；也可使用 FlashAttention 4 或 SageAttention（见 attention-backends.md）。

