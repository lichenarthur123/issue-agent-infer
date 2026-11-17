# DGX Spark 模型栈运维手册

三个 vLLM 实例（OpenAI 兼容 API），Docker Compose 管理，运行在 DGX Spark（GB10，121GB 统一内存）。

| 角色 | 模型 | 端口 | served-model-name |
|---|---|---|---|
| Planner | Qwen3-4B-Instruct-2507-**FP8** | **8001** | `qwen3-4b-instruct-2507` |
| 视觉 | Qwen3.8-27B-FP8（原生多模态，按请求开关 thinking） | **8002** | `qwen3.8-27b-fp8` |
| Grounder | GTA1-7B（Salesforce，定位基准优于 UI-TARS） | **8003** | `gta1-7b`（兼容别名 `ui-tars-1.5-7b`） |

> 注：vision 已于 2026-11-17 从 Qwen3-VL-32B-Instruct-FP8 切换为 Qwen3.8-27B-FP8
> （更新一代、原生多模态、自带可控 thinking、实测解码 ~12 tok/s）。
> 旧权重仍在 models/ 下，回退只需改 compose 两行。

- 镜像：`nvcr.io/nvidia/vllm:26.10-py3`（arm64）
- 编排文件：`deploy/docker-compose.yml`
- 模型权重：`models/`（容器内挂载为 `/models`，只读）
- 下载工具：`.venv/`（huggingface_hub + modelscope）

> **docker 权限**：若命令报 `permission denied ... /var/run/docker.sock`，说明当前会话
> 还没有 docker 组权限。临时方案：命令外层包 `sg docker -c "..."`；永久方案：重新登录桌面。

---

## 1. 启动 / 停止

### 运维脚本（推荐）

`scripts/` 下有两个脚本，封装了下面所有手动命令：

```bash
scripts/svc.sh start              # 全部启动(自动按 先大后小 顺序 + 等待就绪)
scripts/svc.sh start vision       # 启动单个
scripts/svc.sh stop [服务名]      # 停止(默认全部)
scripts/svc.sh restart vision     # 改完 compose 配置后重建
scripts/svc.sh status             # 容器状态 + 健康检查 + 内存
scripts/svc.sh logs vision        # 跟踪日志

scripts/test.sh                   # 三个服务全部冒烟测试
scripts/test.sh grounder          # 只测某个(自带测试图, 自动校验结果)
```

脚本内置 docker 权限自检，未生效时会自动借 `sg` 执行，无需手动处理。

### 手动命令

```bash
cd ~/Desktop/workspace/deploy
```

### 启动（顺序很重要：先大后小，避免显存测量竞争）

```bash
docker compose up -d vision      # 1. 最大的先起 (~63GB)
docker compose up -d grounder    # 2. (~22GB)
docker compose up -d planner     # 3. (~19GB)
```

每个服务加载需要 2~5 分钟，等健康检查通过再启动下一个：

```bash
curl -sf localhost:8002/health && echo vision OK
curl -sf localhost:8003/health && echo grounder OK
curl -sf localhost:8001/health && echo planner OK
```

### 停止 / 重启

```bash
docker compose down                    # 全部停止并删除容器
docker compose stop vision             # 停单个
docker compose up -d --force-recreate vision   # 改完 compose 配置后重建单个
```

### 开机自启（已启用）

三个服务均已配置 `restart: unless-stopped`，并通过
`depends_on: service_healthy` 形成启动链 vision → grounder → planner，
`docker compose up -d` 一条命令即可按序拉起全部。
注意：机器重启后 Docker 会直接拉起三个容器（不经过 compose 依赖排序），
若个别服务因显存竞争启动失败，restart 策略会自动重试直至成功。
另外已配置日志限额（单文件 100MB × 3），防止日志撑爆磁盘。

---

## 2. 日常监控

### 实时监控面板（推荐）

```bash
scripts/monitor.py        # GPU/内存/CPU/网络 一屏实时刷新, Ctrl+C 退出
scripts/monitor.py 2      # 自定义刷新间隔(秒)
```

GPU 行显示利用率/温度/功耗；内存行是统一内存总占用（模型的显存也在里面）；
CPU 显示总负载和 20 个核的分核负载；网络显示各接口实时上下行速率。

### 手动命令

```bash
# 容器状态（Up X minutes (healthy) 即正常）
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# 内存占用（统一内存，关注 available）
free -h

# 某个服务的实时日志
docker logs -f vllm-vision          # Ctrl+C 退出
docker logs --tail 50 vllm-planner  # 只看最后 50 行

# GPU/推理引擎 KV 池使用情况（启动日志里）
docker logs vllm-vision 2>&1 | grep -E "Available KV cache|GPU KV cache size"
```

### 健康检查 / 模型列表

```bash
curl localhost:8001/v1/models
curl localhost:8002/v1/models
curl localhost:8003/v1/models
```

---

## 3. API 冒烟测试

### planner（文本）

```bash
curl localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b-instruct-2507",
  "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
  "max_tokens": 100
}'
```

### vision（图文）

```bash
curl localhost:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-fp8",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "描述这张图"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64图片>"}}
  ]}],
  "max_tokens": 200
}'
```

### vision（工具调用，解析器为 qwen3_xml）

```bash
curl localhost:8002/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b-fp8",
  "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "description": "查询城市天气",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "max_tokens": 500
}'
```

> vision 已开启 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
> （该模型输出 qwen3xml 格式，**不要用 hermes**）和 `--reasoning-parser qwen3`
> （思考过程在响应的 `reasoning` 字段，本版本 vLLM 不用 `reasoning_content`）。
> 关 thinking：请求里加 `"chat_template_kwargs": {"enable_thinking": false}`。

### grounder（必须用 UI-TARS 官方 prompt 格式才会输出坐标）

```bash
curl localhost:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ui-tars-1.5-7b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.\n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\nclick(point=\u0027<point>x1 y1</point>\u0027)\n\n## User Instruction\n点击搜索按钮"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64截图>"}}
  ]}],
  "max_tokens": 200, "temperature": 0.1
}'
```

> 期望输出形如：`Action: click(point='(x,y)')`。如果只用日常语言问它，
> 它只会返回文字描述，不会给坐标——这不是故障。

---

## 4. 内存与上下文调优

统一内存共 121GB，当前分配（compose 里每个服务的 `--gpu-memory-utilization`）：

| 服务 | utilization | 约合 | KV 池 | 当前上下文 |
|---|---|---|---|---|
| vision | 0.40 | ~48GB | 12.76GB（FP8 KV，28.7万 token） | 65536 |
| grounder | 0.20 | ~24GB | ~2GB | 32768 |
| planner | 0.11 | ~13GB | ~7GB（FP8 KV） | 49152 |

**KV 每 token 参考**：vision ≈ 47KB（FP8 KV），planner ≈ 72KB（FP8 KV），grounder ≈ 56KB。
已启用优化：vision/planner `--kv-cache-dtype fp8`（注意：**Qwen2.5-VL 架构的 grounder 不能开 FP8 KV，会输出乱码**）；planner 使用官方 FP8 权重（提速约 2 倍）；prefix caching 默认开启（实测命中率 ~48%）；grounder 已换为 GTA1-7B（ScreenSpot-Pro 50.1 vs UI-TARS-1.5 的 42.0）。

### 单并发加速（2026-11-17 实测调优）

三个服务均按**单并发**场景优化（`--max-num-seqs 8`），并启用投机解码：

| 服务 | 方案 | 效果 |
|---|---|---|
| vision | MTP 投机解码 3 token（模型自带 MTP 头） | 7.5 → **12.0 tok/s（+60%）** |
| planner / grounder | ngram 投机解码（无需草稿模型） | 重复/模板化输出接受率 100%，自由文本空转无开销 |

vision 侧实测结论：MTP=3 最优；MTP=5（10.7 tok/s）和 DFlash2 草稿模型（10.9 tok/s）
反而更慢——草稿过长时验证开销在 GB10 弱算力下不划算。DFlash2 权重保留在
`models/qwen3.8-27b-dflash2/`，切换只需改 `--speculative-config` 一行。
注意：vision 开投机后 Mamba 缓存块数有限，`--max-num-seqs` 不能超过 209
（混合架构每序列独占一块 Mamba 状态缓存）。

### 调大某模型上下文

1. 按上表估算新上下文需要的 KV（tokens × 每 token KV）。
2. 编辑 `deploy/docker-compose.yml`：改 `--max-model-len`，
   并同步上调 `--gpu-memory-utilization`（每 0.01 ≈ 1.2GB）。
3. 重建：`docker compose up -d --force-recreate <服务名>`
4. 启动失败看日志里的 `estimated maximum model length`，按提示回调。

### 系统内存紧张（free -h 的 available < 5GB）

- 关闭 Firefox 等桌面大户（统一内存下它们和模型抢内存）；
- 或下调 grounder/planner 的 utilization 0.01~0.02。

---

## 5. 常见故障

| 症状 | 排查 |
|---|---|
| 容器启动后立刻退出 | `docker logs vllm-<name>` 找 `ValueError: ... KV cache ... larger than available` → KV 不足，按第 4 节调参数 |
| 健康检查一直不通 | `docker logs -f vllm-<name>` 看是否还在加载（32B 要约 4~5 分钟）；出现 `RuntimeError: Engine core initialization failed` 往上翻找根因 |
| 推理速度异常慢 | 确认没开其他吃 GPU 的程序；`docker logs` 看吞吐行 `Avg generation throughput` |
| 端口不通 | `docker ps` 确认端口映射；确认没被其他进程占用：`ss -tlnp \| grep 800` |
| 系统卡死/OOM | 立即 `docker compose stop vision`，然后按第 4 节下调内存 |

---

## 6. 模型与镜像更新

### 重新下载/更新模型（断点续传）

```bash
cd ~/Desktop/workspace
HF_ENDPOINT=https://hf-mirror.com .venv/bin/hf download <组织>/<仓库> --local-dir models/<目录>
```

huggingface.co 直连不通，**必须**带 `HF_ENDPOINT=https://hf-mirror.com`。

### 更新 vLLM 镜像

```bash
docker pull nvcr.io/nvidia/vllm:<新tag>   # tag 列表见 NGC catalog
# 改 deploy/docker-compose.yml 里三处 image，然后逐个 force-recreate
```

---

## 7. 目录结构

```
~/Desktop/workspace/
├── readme.md                  # 本文件（运维手册）
├── deploy/
│   ├── docker-compose.yml     # 服务编排（端口/内存/上下文都在这里改）
│   └── README.md              # 部署说明（与本文档互补）
├── models/
│   ├── qwen3.8-27b-fp8/               (~29GB, vision 使用)
│   ├── qwen3.8-27b-dflash2/           (~3.6GB, DFlash2 草稿, 备选加速)
│   ├── qwen3-4b-instruct-2507-fp8/    (~5GB, planner 使用)
│   ├── qwen3-4b-instruct-2507/        (~8GB, BF16 旧版, 可删)
│   ├── qwen3-vl-32b-instruct-fp8/     (~35GB, 已被 Qwen3.8-27B 替代, 可删)
│   ├── qwen3-vl-32b-thinking-fp8/     (~34GB, 已被 Qwen3.8-27B 替代, 可删)
│   ├── qwen3-vl-8b-instruct/          (~17GB, 备份未部署)
│   ├── gta1-7b/                       (~17GB, grounder 使用)
│   └── ui-tars-1.5-7b/                (~32GB, 已被 GTA1 替代, 可删)
└── .venv/                     # hf/modelscope 下载工具
```
