# 实时服务与 WebSocket 协议

- 服务文件：`release_server.py`
- 框架：FastAPI + Uvicorn；WebSocket 用于帧流式传输
- 模型加载：应用 `lifespan` 中完成，统一复用

## 启动与环境

- 环境变量：
  - `MODEL_FOLDER`：Wan 模型根目录（见 `settings.py`）
  - `CONFIG`：配置文件路径，默认 `configs/self_forcing_server_14b.yaml`
  - `DO_COMPILE`：`true/false`，是否对模型进行 `torch.compile`
- 启动：`uvicorn release_server:app --host 0.0.0.0 --port 8000`

## HTTP 端点

- `GET /health`：存活检查
- `GET /`：简易演示页面（`templates/release_demo.html`）
- `POST /upload_video`：上传视频，返回临时路径，用于 V2V
- `POST /upload_start_frame`：上传首帧图片，用于 I2V/视频续写
- `GET /download_video/{session_id}`：下载已生成的 MP4（服务端拼接缓存帧）

## WebSocket 会话

- 端点：`/session/{id}`
- 握手：服务器先发 `{"status": "ready", "worker": <hostname>}`
- 客户端发送参数与数据：
  - 初始 JSON/MsgPack（可选）参数：
    - `prompt`、`seed`、`num_blocks`、`width/height`、`kv_cache_num_frames` 等
  - 帧输入（可选）：
    - 图片二进制（JPEG），用于 Webcam/V2V 逐帧注入；字段 `image`
    - 同时可携带 `strength`（0-1）与 `request_id` 用于溯源

### 服务端生成循环

- `GenerationSession.generate_block(...)`：每个块完成后触发回调 `frame_callback(pixels, frame_ids, event)`
- 回调：
  - 等待 CUDA 事件，GPU->CPU 异步拷贝
  - JPEG 编码（或 MsgPack 打包）
  - 通过 WebSocket 发送给客户端；同时在内存暂存以支持下载
- 结束：
  - 生成完成或被取消时，发送 `{"session_id": id, "status": "completed"}`

#### 全链路并行流水线（深度）

- 上传流：`upload_stream = torch.cuda.Stream()`，将客户端图片帧从 CPU（pin memory）异步拷入 GPU 并标准化到 `[-1,1]`（`release_server.py:657-666`）。
- 生成流：主线程在 `generate_pool`（单线程池）中运行 `GenerationSession.generate_block_internal`，逐步执行扩散与 KV 刷新、VAE 解码。
- 下载流：`download_stream = torch.cuda.Stream()`，回调中用事件 `event.record()` 标记 GPU 计算完成；随后将像素异步拷回 CPU（`release_server.py:727-733`）。
- 编码/发送：`encode_pool`（24 线程）在 CPU 上做 JPEG 编码或 MsgPack 打包并通过 WebSocket 发送；同时将帧副本写入 `session_frames_storage` 以支持下载拼接（`release_server.py:978-1013`）。

伪代码（发送侧）：

```
on frame_callback(pixels [1,T,3,H,W], frame_ids, event):
  event -> download_stream.wait_event()
  cpu_tensor = zeros_like(pixels, pin_memory=True)
  with download_stream: cpu_tensor.copy_(pixels)
  normalized = (cpu_tensor + 1) * 0.5
  enqueue encode_pool jobs: for each i in T -> JPEG(normalized[i]) -> ws.send(bytes)
  append normalized clone to session_frames_storage[session_id]
```

#### GenerationSession 关键字段

- 尺寸：`width/height` 对齐 8，潜空间 `(latent_height=H/8, latent_width=W/8)`（`release_server.py:365-371`）。
- 噪声与潜空间：
  - `all_latents ∈ [1, num_blocks*3, 16, H/8, W/8]`
  - `noise` 同形，按 `seed` 初始化（`release_server.py:402-404`）。
- 步长：`denoising_step_list = get_denoising_schedule(...)` 动态依 `strength` 生成（`release_server.py:416-418`）。
- VAE 缓存：`decode_vae_cache` 长度约 55，跨块复用（`release_server.py:394-396`）。
- 上下文缓冲：`frame_context_cache` 维护用于 KV 刷新所需的最近帧，长度 `1+(kv_cache_num_frames-1)*4`（`release_server.py:388-389`）。

#### `generate_block_internal` 的时序

1) 准备 `noisy_input`：从 `noise` 中切出当前块对应的 3 帧；若启用 `start_frame/input_video`，则使用编码潜空间按强度线性混合（`release_server.py:421-431`）。
2) 时间循环：对 `denoising_step_list` 从大到小遍历：
   - 前 K-1 步：调用 `models.transformer(..., kv_cache, crossattn_cache)` 得到 `pred_x0`，用 `scheduler.add_noise(pred_x0, randn, next_t)` 进行 re-noise（`release_server.py:678-694`）。
   - 最后一步：直接取 `pred_x0` 作为该块结果，写入 `all_latents`。
3) 解码：在编译 stance（分辨率匹配时）下调用 `vae_decoder(denoised_pred.half(), *decode_vae_cache)`，得到像素帧并更新缓存（`release_server.py:712-719`）。
4) 跳帧：首块跳过前 3 帧（因为 KV 刷新会用到），仅将后续帧发送（`release_server.py:721-724`）。
5) 回调：创建 CUDA 事件 `event.record()`，提交给 `frame_callback`；随后更新块进度与计数器。
6) KV 刷新：立即以“干净上下文”再次执行一次（见因果推理文档），确保下一块可见最新的干净历史。

#### KV 刷新实现细节

- 入口：`recompute_kv_cache(models)` 会在每块开始时被调用（`release_server.py:614-650`）。
- 构造 `clean_context_frames`（`get_clean_context_frames`）：
  - 若 `keep_first_frame=true` 或上下文不足 `kv_cache_num_frames`，使用首帧 + 最近 `kv_cache_num_frames-1` 帧；
  - 否则，将首帧从缓存像素重编码为潜空间并与最近 `kv_cache_num_frames-1` 帧拼接，避免长上下文导致显存过大（`release_server.py:560-579`）。
- 为上下文构造专用 `block_mask`（仅覆盖上下文长度），并以 `t=0` 前向一次，写入 KV 缓存；然后清空 `block_mask` 恢复正常生成（`release_server.py:632-648`）。

备注：
- `init_models(...)` 中计算了 `attn_size = kv_cache_num_frames + num_frame_per_block` 并赋给 `pipeline.local_attn_size`，随后将各 Block 的 `local_attn_size` 置 `-1`（全局注意力），实际有效窗口由 `block_mask` 与上下文长度共同约束（`release_server.py:540-556`）。

#### JPEG/MsgPack 编码与协议

- 默认编码 JPEG（质量 90）；若 `?fmt=msgpack`，则以 MsgPack 返回 `{image: <bytes>, request_id}`，方便前端对齐异步帧与请求来源（`release_server.py:948-973`）。

#### 错误恢复与资源清理

- 任一阶段异常会调用 `session.dispose()` 终止会话；`frame_sender_task`/`generate_task` 在 `finally` 中取消；完成时回传 `completed` 并支持下载（`release_server.py:1066-1083`）。

### 参数热更新与 Prompt 插值

- 客户端可在会话中变更 `prompt`，服务端对嵌入做线性插值（`interp_steps`）平滑过渡。
- 可动态变更 `seed`、发送新图像帧等；发送 `{action: "reset"}` 可重置会话状态。

## Webcam / V2V

- Webcam：
  - 首块编码 9 帧，后续每块 12 帧，保证与 3 帧生成块对齐
  - `v2v.encode_video_latent(..., stream=True)` 做流式编码
- V2V：
  - `v2v.load_video_as_rgb` 以 16fps 重采样（过长视频自动重采样），支持 URL/本地
  - 编码为潜空间后，与噪声按 `strength` 线性混合确定起始状态

## 资源与容错

- 大量使用 `torch.cuda.Stream` 与事件实现上传/下载/推送并行化
- 出错时尽量释放会话资源（`session.dispose()`），回收缓存

### 服务器多工

- 模型加载一次后复用；若需要多 GPU，可通过多进程/多实例绑定不同 `CUDA_VISIBLE_DEVICES`，或复制 `Models` 到指定 GPU（`copy_models(...)`）。
- 若需要多路并发同 GPU：可为每个会话新建 `GenerationSession`，注意 KV/跨注意力缓存为每会话一套，避免互扰。

## 客户端示例

- `test_client.py`：简单 WebSocket 客户端，循环发送同一图片帧并接收返回帧
- `test_request.py`：HTTP 上传/下载交互示例
