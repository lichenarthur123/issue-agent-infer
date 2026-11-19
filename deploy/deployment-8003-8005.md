# 本地 Agent 模型栈部署文档（8003 / 8005）

> 更新日期：2026-11-19
> 部署主机：NVIDIA DGX Spark（GB10，121GB 统一内存，arm64）

## 1. 服务总览

| 端口 | 服务角色 | 模型 ID（调用时填这个） | 模型 | 能力 |
|---|---|---|---|---|
| 8005 | 视觉主模型 | `qwen3.8-27b-nvfp4` | Qwen3.8-27B（NVFP4/FP8 混合量化，原生多模态） | 文本对话、图像理解、function calling |
| 8003 | Grounder | `gta1-7b`（别名 `ui-tars-1.5-7b`） | GTA1-7B（Qwen2.5-VL 架构，BF16） | GUI 截图坐标 grounding，输出如 `click(point='x y')` |

两个服务均为 **OpenAI 兼容 API**（`/v1/chat/completions`、`/v1/models`、`/health`），**无鉴权**，仅限内网使用。

### 访问地址

| 入口 | 8005（vision） | 8003（grounder） |
|---|---|---|
| 本机 | `http://localhost:8005` | `http://localhost:8003` |
| Mac 网线直连（静态，推荐） | `http://10.10.10.1:8005` | `http://10.10.10.1:8003` |
| WiFi 网段（DHCP，可能变化） | `http://10.213.114.86:8005` | `http://10.213.114.86:8003` |

服务绑定 `0.0.0.0`，所有网卡可达。

## 2. 部署方式

- 部署形态：Docker Compose（compose 文件位于部署机 `deploy/docker-compose.yml`）
- 镜像：`nvcr.io/nvidia/vllm:26.10-py3`（arm64）
- 模型以本地目录只读挂载进容器（宿主机 `models/` → 容器内 `/models`）
- 两个实例共享 GPU 统一内存，通过 `--gpu-memory-utilization` 划分；
  `restart: unless-stopped` + `depends_on(service_healthy)` 链式依赖（vision-fp4 先起，grounder 后起），开机自动拉起
- 启动：`cd deploy && docker compose up -d`；停止：`docker compose down`
- 健康检查：`curl -sf http://localhost:8005/health`、`curl -sf http://localhost:8003/health`

## 3. 8005 视觉模型（vision-fp4）

### 3.1 部署参数（完整启动命令）

```bash
vllm serve /models/qwen3.8-27b-nvfp4 \
  --served-model-name qwen3.8-27b-nvfp4 \
  --port 8000 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --linear-backend auto \
  --kernel-config '{"linear_backend_per_quant":{"mixed_precision":"b12x","nvfp4_w4a4":"b12x","nvfp4_w4a16":"b12x"}}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"dspark","model":"/models/qwen3.8-27b-dspark-fp8-dynamic","num_speculative_tokens":7}' \
  --max-num-seqs 8 \
  --enable-prompt-tokens-details
```

参数说明：

| 参数 | 取值 | 说明 |
|---|---|---|
| gpu-memory-utilization | 0.40（约 48GB） | 权重约 22GB + FP8 KV 缓存约 12.8GB |
| max-model-len | 65536 | 上下文上限（输入+输出） |
| kv-cache-dtype | fp8 | KV 缓存 FP8 量化 |
| 投机解码 | DSpark k=7 | 草稿模型 `qwen3.8-27b-dspark-fp8-dynamic`（5 层 FP8 草稿头），实测 decode 33.2 tok/s，比原 MTP 方案快约 25% |
| tool-call-parser | qwen3_xml | Qwen3.8 输出 qwen3xml 格式，必须用此解析器 |
| reasoning-parser | qwen3 | 思考内容分离到 `reasoning_content` 字段 |
| max-num-seqs | 8 | **按单并发场景设计**，客户端请串行调用，至多 2 路并发 |

### 3.2 模型参数

| 项 | 值 |
|---|---|
| 架构 | Qwen3.5（`Qwen3_5ForConditionalGeneration`，混合线性注意力架构，原生多模态） |
| 层数 | 64 层（3 层 linear attention : 1 层 full attention 交替） |
| hidden size | 5120，attention heads 24（KV heads 4），head_dim 256 |
| 视觉编码器 | 27 层，hidden 1152，patch size 16 |
| 量化 | MLP 层 NVFP4（W4，group size 16）+ 注意力投影/lm_head FP8（compressed-tensors 格式） |
| 原生精度 | bfloat16 |

### 3.3 接口示例

文本对话：

```bash
curl http://10.10.10.1:8005/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-nvfp4",
  "messages": [{"role": "user", "content": "你好"}],
  "max_tokens": 256
}'
```

图像理解（base64 data URL，支持 PNG/JPEG，截图建议不超过 1920×1080）：

```bash
curl http://10.10.10.1:8005/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-nvfp4",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
    {"type": "text", "text": "描述这张图"}
  ]}]
}'
```

其他约定：

- 流式：加 `"stream": true`，SSE 返回
- Thinking：默认开启，思考内容在 `reasoning_content`，正式答案在 `content`；
  关闭加 `"chat_template_kwargs": {"enable_thinking": false}`
- Function calling：走 OpenAI 标准 `tools` 协议（`tool_choice: "auto"`）
- 性能参考：TTFT 约 0.3~1.1s（视 prompt 长度），decode 约 33 tok/s；请求超时建议 ≥180s

## 4. 8003 Grounder（gta1-7b）

### 4.1 部署参数（完整启动命令）

```bash
vllm serve /models/gta1-7b \
  --served-model-name gta1-7b ui-tars-1.5-7b \
  --port 8000 \
  --gpu-memory-utilization 0.20 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1}'
```

参数说明：

| 参数 | 取值 | 说明 |
|---|---|---|
| gpu-memory-utilization | 0.20（约 24GB） | BF16 权重约 17GB + KV 缓存 |
| max-model-len | 32768 | 上下文上限 |
| 投机解码 | ngram | 重复输出场景接受率高 |
| served-model-name | 两个名字都可用 | `gta1-7b` 或 `ui-tars-1.5-7b`（UI-TARS 兼容别名） |

注意：该模型是 Qwen2.5-VL 架构，**不能用 `--kv-cache-dtype fp8`**（会输出乱码），因此 KV 为 BF16。

### 4.2 模型参数

| 项 | 值 |
|---|---|
| 架构 | Qwen2.5-VL（`Qwen2_5_VLForConditionalGeneration`） |
| 规模 | 7B，28 层，hidden 3584，attention heads 28（KV heads 4） |
| 视觉编码器 | 32 层，hidden 1280，patch size 14，window attention（window 112，4 层全注意力） |
| 精度 | bfloat16（无量化） |
| 训练来源 | HelloKKMe/GTA1-7B（UI-TARS-1.5 同类 GUI grounding 模型） |

### 4.3 接口示例

```bash
curl http://10.10.10.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gta1-7b",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<截图BASE64>"}},
    {"type": "text", "text": "找到\"确定\"按钮的位置"}
  ]}],
  "max_tokens": 128,
  "temperature": 0
}'
```

返回形如 `click(point='540 960')` 的动作文本。grounding 质量取决于输入截图分辨率。

## 5. 注意事项

- 两服务均为**单并发优化**配置（max-num-seqs 8），并发过高会互相拖慢，调用方请串行或低并发
- 无鉴权，不要直接暴露到公网；如需对外，先加网关鉴权或防火墙白名单
- 8001 / 8002 / 8004 端口的旧服务已下线，不要调用
- OOM 处理：优先降 `--max-model-len` 或对应实例的 `--gpu-memory-utilization`
