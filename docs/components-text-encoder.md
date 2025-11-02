# 文本编码器（UMT5-XXL）

- 封装：`utils/wan_wrapper.WanTextEncoder`
- 模型：`wan/modules/t5.umt5_xxl`（encoder-only），Tokenizer 使用 `HuggingfaceTokenizer`
- 权重路径：`${MODEL_FOLDER}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors`

## 前向接口

- 输入：`List[str] text_prompts`
- 输出：`{"prompt_embeds": Tensor}`
  - `ids, mask = tokenizer(prompts, return_mask=True)`
  - `context = text_encoder(ids, mask)`，并按 `mask` 将 padding 位置置零
  - 结果张量形状：`[batch, seq_len<=512, 4096]`

## 设备与精度

- 默认在 CUDA 上推理，`bfloat16` 精度。
- 在低显存模式下，可配合 `demo_utils/memory.move_model_to_device_with_memory_preservation` 动态迁移。

## 交互位置

- 因果推理时，文本特征作为跨注意力 `context` 输入至 `CausalWanAttentionBlock`。
- 跨注意力缓存 `{k, v, is_init}` 会在首次使用后复用，避免重复编码。

### Token 化与序列长度

- Tokenizer 固定 `seq_len=512`，会自动截断或 pad；`mask.gt(0).sum(dim=1)` 得到每条样本的有效长度并在编码后将 padding 段置 0（`utils/wan_wrapper.py:49-55`）。
- 在跨注意力中，文本序列再经 `text_embedding: Linear(4096→dim)×2 + GELU` 转换到与视觉分支一致的维度（`wan/modules/causal_model.py:192-216` 的上下文路径）。

### Prompt 插值（Streaming 内热更新）

- 会话内变更 Prompt 时，服务端会取当前 `prompt_embeds` 与新 Prompt 的编码做线性插值，分若干步替换到后续块（`release_server.py:590-606`）。这可以在不中断推理的情况下平滑改变语义。
