# 因果推理管线（Self-Forcing Inference）

- 文件：`pipeline/causal_inference.py`
- 入口：`CausalInferencePipeline.inference(...)`
- 上层：`release_server.GenerationSession.generate_block_internal(...)`

## 目标

- 将扩散模型转化为自回归生成：逐块输出潜空间帧，刷新 KV Cache，支持流式播放与后续块条件化。

## 初始化

- 加载模型与配置：`num_frame_per_block`、`denoising_step_list`、`independent_first_frame`、`local_attn_size`。
- 构造缓存：
  - `kv_cache1`：长度等于 Block 数（通常 30），每项包含 `k/v` 与 `*_end_index`。
  - `crossattn_cache`：同样长度，缓存文本跨注意力的 `k/v` 与 `is_init` 标记。
- 帧内序列长度固定：`frame_seq_length = 1560`。

## 推理步骤

- 输入：
  - `noise`：形状 `[B, T, 16, H/8, W/8]` 的初始噪声或混合潜空间（V2V 时与编码潜空间线性混合）。
  - `text_prompts`：字符串列表。
  - `initial_latent`（可选）：用于 I2V 或视频扩展。
- 输出：像素帧 `video`（或同时返回潜空间 `latents`）。

- 过程：
  1) 文本编码：`conditional_dict = text_encoder(text_prompts)`
  2) 缓存清零/重用：重置 KV/LN 指针与跨注意力 `is_init` 标志。
  3) 上下文缓存：若 `initial_latent` 存在，以 `t=0`（或 `context_noise`）将参考帧写入缓存。
  4) 时间/空间双循环：
     - 外层按当前块帧数（通常 3）循环；`independent_first_frame` 时首帧为单帧块。
     - 内层按 `denoising_step_list` 从大到小时间步：
       - 调用 `WanDiffusionWrapper(..., kv_cache, crossattn_cache, current_start)` 得到 `(flow_pred, pred_x0)`。
       - 若非最后一步：用 `scheduler.add_noise(pred_x0, randn, next_t)` 重新扰动，继续下一步。
       - 最后一步：将 `pred_x0` 写入输出与 `all_latents`。
  5) KV 刷新：以近零时间步 `context_timestep` 对已生成段再次前向，保证“干净上下文”填充进 KV Cache（可裁剪长度）。
  6) VAE 解码：`vae.decode_to_pixel(output)`，并规范到 `[0,1]`。

### 伪代码（单块，独立首帧关闭情形）

```
current_start_frame = ...
noisy_input = noise[:, current_start_frame:current_start_frame+3]
for idx, t in enumerate(denoising_step_list):
  timestep = t * ones([B, 3])
  _, pred_x0 = generator(
      noisy_image_or_video=noisy_input,
      conditional_dict=conditional_dict,
      timestep=timestep,
      kv_cache=kv_cache1,
      crossattn_cache=crossattn_cache,
      current_start=current_start_frame * frame_seq_length)
  if idx < K-1:
      next_t = denoising_step_list[idx+1]
      noisy_input = scheduler.add_noise(
         pred_x0.flatten(0,1), randn_like(...), next_t*ones(B*3)).unflatten(0, pred_x0.shape[:2])
output[:, current_start_frame:current_start_frame+3] = pred_x0

# Clean context refresh
clean_context_frames = output[:, :current_start_frame+3]
if max_num_context_frames:
  clean_context_frames = tail(clean_context_frames, max_num_context_frames)
context_timestep = ones([B, clean_context_frames.shape[1]]) * context_noise
clean_context_frames = scheduler.add_noise(clean_context_frames.flatten(0,1), randn_like(...), context_timestep*ones(...)).unflatten(...)
generator(noisy_image_or_video=clean_context_frames, timestep=context_timestep, kv_cache=kv_cache1, crossattn_cache=crossattn_cache, current_start=0)
```

### FlowMatch 与 x0/flow 的转换

- 训练/推理统一采用 FlowMatch 视角：
  - `x_t = (1-σ_t) x_0 + σ_t ε`
  - 模型预测 `flow_pred ≈ ε - x_0`
  - 由此得到：`x0_pred = x_t - σ_t * flow_pred`（`utils/wan_wrapper.py:181-205`）。
- re-noise：使用 `FlowMatchScheduler.add_noise` 将 `x0_pred` 重投影回下一步 `x_t`（`utils/scheduler.py:174-201`）。

## 步长与时间映射

- `utils/scheduler.FlowMatchScheduler` 定义 `sigmas ∈ [sigma_min, sigma_max]` 的线性表，并映射到 `timesteps`。
- `v2v.get_denoising_schedule` 根据 `strength∈[0,1]` 在 1000 步上均匀采样，再投影回训练时刻。
- `warp_denoising_step` 为兼容不同训练刻度的“扭曲”映射（见配置）。

## 形状与显存要点

- KV 长度与显存：`kv_cache_size ≈ local_attn_size*1560` 或默认全局上限（约 32760）。
- 每个 Block 保存一份 KV/CrossAttn 缓存；14B 下单卡 KV 可达 25GB 量级，需结合 `kv_cache_num_frames`/`local_attn_size` 控制。
- `num_frame_per_block` 影响块掩码与缓存刷新频率，取 3 是实时与质量的平衡点。

### 长视频与滑窗（cache_start/current_start）

- 在扩展/长视频场景，可设置 `cache_start` 指定 KV 的全局对齐位置，使模型对齐历史窗口；`current_start` 指示当前块的全局 token 起点（`wan/modules/causal_model.py:224-228`）。
- 结合 `local_attn_size` 可实现固定窗口长度的自回归滚动。

## 与服务层配合

- `GenerationSession` 在每个块结束时触发解码与回调；在 `start_frame` 或 `input_video` 模式下，会先通过 VAE 编码建立初始潜空间并按 `strength` 混合噪声。
- 支持 Prompt 插值：会话内替换 Prompt 时可线性插值数步以平滑过渡。
