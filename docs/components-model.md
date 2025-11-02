# 扩散主干与 CausalWanModel

本节聚焦模型内部：Patch Embedding、因果自注意力、跨注意力、KV/CrossAttn 缓存、时间嵌入与输出头。

- 入口封装：`utils/wan_wrapper.WanDiffusionWrapper`
- 具体实现：`wan/modules/causal_model.py`（因果版），`wan/modules/model.py`（基础版）

## 包装器 WanDiffusionWrapper

- 作用：统一加载预训练 `Wan` 模型，选择是否因果版（`is_causal=True`），并挂载 FlowMatch 调度器。
- 关键点：
  - `uniform_timestep`：非因果扩散时所有帧共享同一 `timestep`。
  - `get_scheduler()`：返回实现 `SchedulerInterface` 的 FlowMatch 调度器并注入若干转换方法（x0<->noise 等）。
  - 预测转换：`_convert_flow_pred_to_x0` 将流匹配预测转换为 x0；训练/采样中反向转换也可用。
  - `fuse_projections()`：将 q/k/v 线性层融合为单层 `to_qkv`，降低开销。

## CausalWanModel 结构

- Patch Embedding：`Conv3d(in_dim=16, dim)`，时空三维打补丁，步长为 `(1,2,2)` 对应潜空间下采样。
- 时间嵌入：`time_embedding` + `time_projection` 生成 6 路调制向量，参与 Block 的自注意力/FFN 调制。
- Block（重复 `num_layers` 次）：`CausalWanAttentionBlock`
  - 自注意力：`CausalWanSelfAttention`，支持：
    - RoPE（3D 拆分：帧/高/宽），`rope_apply` 应用到 q/k。
    - Flex Attention：`torch.nn.attention.flex_attention` 配合 `BlockMask` 实现分块因果可见性与右侧填充。
    - KV Cache：每步写入当前帧的 K/V；`global_end_index`/`local_end_index` 用于增量更新与窗口控制。
    - 可选局部注意力：`local_attn_size>0` 时仅允许回看最近若干帧（`≈ local_attn_size*1560` tokens）。
  - 跨注意力：对文本上下文进行一次交互（可选归一化），并支持跨注意力缓存 `{k,v,is_init}`。
  - FFN：`GELU` 两层 MLP，受时间调制因子作用。
- Head：将序列恢复为 `[C_out=16, F, H/8, W/8]`，对应潜空间维度。
- Unpatchify：逐样本依据原网格尺寸恢复潜空间 3D 帧张量。

## 注意力掩码与序列长度

- 帧内序列长度：`frame_seq_length = 1560`（对应 H/8×W/8×patch 内 token 数）。
- 整体序列长度：`num_frames * frame_seq_length`，对齐到 128 的倍数用于 Flex Attention 的块编排。
- 块掩码 `get_block_mask(...)`：确保每一块仅可见到当前块末端的 token，匹配自回归因果假设。

### 1560 的来历（以 832×480 为例）

- VAE 解码输出的潜空间分辨率为 `(H/8, W/8)`，即 `480/8=60`、`832/8=104`。
- Patch Embedding 的步长为 `(1,2,2)`，进一步在空间维度每 2×2 聚合一个 token；
- 因此每帧 token 数为 `(60/2) * (104/2) = 30 * 52 = 1560`，与代码常量一致（`pipeline/causal_inference.py:35`）。

### BlockMask 的构造与填充

- 为了匹配 Flex Attention 的块式并行，序列长度右侧补零到 128 的倍数（`wan/modules/causal_model.py:116-118`）。
- 使用 `ends` 向量标出每个查询位置可见的最右端界（当前块末尾），以此在 `create_block_mask` 中生成 `BlockMask`（`wan/modules/causal_model.py:109-141`）。
- Query/Key/Value 在进入 Flex Attention 前，先将 RoPE 后的 Q/K 与 V 也做同样长度的右填充，并在输出后裁掉填充部分（`wan/modules/causal_model.py:262-268` 附近）。

## 前向接口（推理）

- 输入：
  - `x`：按帧拆分后的潜空间列表，内部拼接为 `[B*F, S, C]` 运行。
  - `t`：`[B]` 或 `[B,F]` 的时间步（因果版通常是逐帧步长）。
  - `context`：文本特征列表，统一补零至固定长度 512。
  - `kv_cache`/`crossattn_cache`：长度等于 Block 数的字典列表。
  - `current_start`/`cache_start`：全局/缓存偏移，支持长视频滑窗。
- 输出：`[B, F, 16, H/8, W/8]`（x0 预测，包装层已转换）。

### KV/CrossAttn 缓存的语义与更新

- 形状：
  - KV：`k/v ∈ [B, KV_LEN, num_heads, head_dim]`；`KV_LEN = local_attn_size*1560` 或全局上限（训练/服务不同文件中默认上限约 32760）。
  - CrossAttn：`k/v ∈ [B, 512, num_heads, head_dim]`，`is_init` 表示是否已初始化本块的文本上下文键值。
- 指针：
  - `local_end_index`：本次写入后局部末端位置（等于当前帧 token 数累加）。
  - `global_end_index`：等价于 `local_end_index` 的别名，用于重置/兼容。
- 更新：每个时间步写入当前帧的 K/V；当 re-noise 继续时仍覆盖缓存的尾部（因 Query 总是在「最新截至点」内）。
- 刷新上下文：每个块结束后，以“干净上下文”（近零时间步）对已生成帧再跑一遍模型，确保缓存里的上下文与真实 x0 对齐，从而下一块可见的是更干净的历史（见因果推理文档）。

### Q/K/V 融合与 RoPE 三分法

- `fuse_projections()` 将 `q/k/v` 三个线性层拼接为单层 `to_qkv`，减少 Kernel 启动开销（`wan/modules/causal_model.py:203-216`）。
- RoPE 将通道维三等分为（时间/高度/宽度）三组频率，分别应用至对应轴，再拼接回去（`wan/modules/causal_model.py:143-171`）。

## 代码参考

- 封装与调度：`utils/wan_wrapper.py:1`
- 因果注意力：`wan/modules/causal_model.py:1`
- Flex Attention 块掩码：`wan/modules/causal_model.py:140`

更多参考：
- KV 长度与显存：`wan/modules/causal_model.py:192` 中 `max_attention_size` 的计算；
- 调度器与 x0/flow 转换：`utils/wan_wrapper.py:181`, `utils/scheduler.py:97` 及 `utils/scheduler.py:145`。
