# 注意力后端与编译/量化

## 注意力后端

- PyTorch Flex Attention：默认路径，`wan/modules/causal_model.py` 中 `flex_attention` 已 `torch.compile`。
- FlashAttention 4：适配 B200，`pip install flash_attn --no-build-isolation`
- SageAttention 2.2.1：兼容 H100/RTX 5xxx，仓库提供了对应 whl：`libs/sageattention-2.2.1-*.whl`
  - 安装脚本：`bash install_sage.sh`
  - 注意：SageAttention 2++/3 未验证，质量可能下降

## 编译（torch.compile）

- 环境变量 `DO_COMPILE=true` 时：
  - `models.vae_decoder = torch.compile(..., fullgraph=True)`
  - `models.transformer = torch.compile(models.transformer)`
- 分辨率与形状：`release_server` 在 `(832,480)` 时启用默认 stance；其他分辨率将强制 eager 以规避 Dynamo 图稳定性问题。

实现细节：
- CausalWanModel 使用 Flex Attention 本身已被 `torch.compile` 为 `max-autotune-no-cudagraphs`（`wan/modules/causal_model.py:23-24`）。
- 运行时，`compile_models(...)` 对 `vae_decoder` 使用 `fullgraph=True`，对 `transformer` 使用默认编译，避免过多图分裂（`release_server.py:753-756`）。

## 量化（FP8）

- 配置 `enable_fp8=true` 时使用 `torchao.quantization`：
  - `Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())`
  - 仅 Transformer 应用；对质量略有影响，换取显存与吞吐。
