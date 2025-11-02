# Krea Realtime 14B 项目文档（中文）

本目录汇总对仓库各组件与推理流程的深度解读，帮助你快速理解并高效使用该实时 14B 文生视频系统。

- 适用对象：研发、算法工程师、应用集成同学
- 关注重点：在线流式推理、KV Cache 策略、VAE 流式解码、调度器与步长、资源优化

## 文档导航

- 概览与架构
  - [整体架构与数据流](architecture.md)
- 核心组件
  - [扩散主干与 CausalWanModel](components-model.md)
  - [文本编码器（UMT5-XXL）](components-text-encoder.md)
  - [视频 VAE 与流式缓存](components-vae.md)
- 推理与服务
  - [因果推理管线（Self-Forcing Inference）](inference-causal.md)
  - [实时服务与 WebSocket 协议](realtime-server.md)
  - [视频到视频（V2V）与降噪日程](v2v.md)
- 配置与性能
  - [配置项与环境变量说明](config.md)
  - [注意力后端与编译/量化](attention-backends.md)
  - [性能优化与显存策略](performance.md)
- 训练与扩展
  - [Self-Forcing 训练与一致性回放](training-self-forcing.md)
  - [训练核心总览（Self‑Forcing / DMD / SiD / GAN / ODE）](training-core.md)
- 常见问题
  - [FAQ / 故障排查](faq.md)

## 关联代码与入口

- 实时服务入口：`release_server.py`
- 离线采样示例：`sample.py`
- 因果推理管线：`pipeline/causal_inference.py`
- 模型封装与调度器：`utils/wan_wrapper.py`, `utils/scheduler.py`
- Wan 模型与因果注意力：`wan/modules/causal_model.py`, `wan/modules/model.py`
- 视频 VAE 封装：`demo_utils/vae_block3.py`, `wan/modules/vae.py`
- V2V 工具：`v2v.py`

> 建议从「整体架构与数据流」开始阅读，然后根据使用场景深入对应章节。
