# DGX Spark 本地 Agent 模型栈

三个 vLLM 实例（OpenAI 兼容 API），跑在 DGX Spark (GB10, 121GB 统一内存) 上。

| 角色 | 模型 | 端口 | served-model-name |
|---|---|---|---|
| Planner | Qwen/Qwen3-4B-Instruct-2507-FP8 | 8001 | `qwen3-4b-instruct-2507` |
| 视觉模型 | Qwen/Qwen3.8-27B-FP8（原生多模态，可控 thinking） | 8002 | `qwen3.8-27b-fp8` |
| Grounder | HelloKKMe/GTA1-7B | 8003 | `gta1-7b`（别名 `ui-tars-1.5-7b`） |

镜像：`nvcr.io/nvidia/vllm:26.10-py3`（arm64，NGC 直连可拉）
模型目录：`../models/`（compose 里挂载为容器内 `/models`，只读）

## 内存预算（统一内存 121GB）

| 实例 | gpu-memory-utilization | 约合 | 内容 |
|---|---|---|---|
| vision | 0.40 | ~48GB | FP8 权重 ~29GB + KV 12.76GB(65536, FP8 KV) |
| grounder | 0.20 | ~24GB | GTA1-7B BF16 权重 ~17GB + KV(32768) |
| planner | 0.11 | ~13GB | FP8 权重 ~5GB + KV(49152, FP8 KV) |

合计 ~85GB，给桌面/系统预留 ~35GB。如需更长的 context，先加 vision 的
`--max-model-len`（KV 每 token ~47KB），再相应上调 utilization。

## 单并发优化（2026-11-17）

- 全部 `--max-num-seqs 8`（单并发场景，vision 侧上限 209，受 Mamba 缓存块数限制）
- vision：MTP 投机解码 3 token（实测 7.5 → 12.0 tok/s；5 token 或 DFlash2 反而更慢）
- planner / grounder：ngram 投机解码（重复输出接受率 100%）
- vision 工具调用：`--enable-auto-tool-choice --tool-call-parser qwen3_xml`
  （Qwen3.8 输出 qwen3xml 格式，hermes 解析不了）
- vision 推理分离：`--reasoning-parser qwen3`，响应里思考在 `reasoning` 字段；
  请求加 `"chat_template_kwargs": {"enable_thinking": false}` 可关 thinking

## 启动（顺序很重要：先大后小）

```bash
cd deploy
docker compose up -d        # compose 会按 vision → grounder → planner 依赖链顺序拉起
```

三个服务已配置 restart: unless-stopped + depends_on(service_healthy) 链式依赖，
开机后 Docker 会自动拉起（失败会自动重试）。手动单启某个服务仍然可用
`docker compose up -d <服务名>`，首次部署建议用 scripts/svc.sh start 按顺序启动。

## 停止 / 重启

```bash
docker compose down              # 全部停止
docker compose restart vision    # 重启单个
```

## 验证

```bash
# 健康检查
curl localhost:8001/v1/models
curl localhost:8002/v1/models
curl localhost:8003/v1/models

# planner 文本对话
curl localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b-instruct-2507",
  "messages": [{"role": "user", "content": "用一句话介绍你自己"}]
}'

# vision / grounder 图文（把 base64 图片放进 image_url）
curl localhost:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-fp8",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "描述这张图"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}}
  ]}]
}'
```

## 常见问题

- **OOM / CUDA out of memory**：先降 `vision` 的 `--max-model-len`（如 32768），
  或下调对应实例的 `--gpu-memory-utilization`。桌面应用（Firefox 等）也吃统一内存。
- **吞吐异常低**：`docker logs vllm-vision` 看是否走了 fallback kernel；
  确认镜像是 26.10 或更新（早期 Spark 镜像对 Qwen3-VL 有性能问题）。
- **UI-TARS 坐标不准**：grounding 质量取决于输入截图分辨率，属于 agent 框架侧
  预处理问题，不是 vLLM 的问题；参考 UI-TARS 官方推理代码的 resize 逻辑。
- **模型下载**：用 `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 直连不通）。
