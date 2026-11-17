#!/usr/bin/env bash
# 推理服务冒烟测试
# 用法: test.sh [vision|grounder|planner|all]
set -euo pipefail
target="${1:-all}"
IMG=/tmp/svc_smoke_test.png

gen_image() {
  # 生成测试图: 640x480, 左上红 / 中间绿 / 右下蓝
  python3 - <<'PYEOF'
import zlib, struct
W,H=640,480
def px(x,y):
    if x<213 and y<160: return (255,0,0)
    if x>=427 and y>=320: return (0,0,255)
    if 213<=x<427 and 160<=y<320: return (0,200,0)
    return (240,240,240)
raw=b''.join(b'\x00'+b''.join(bytes(px(x,y)) for x in range(W)) for y in range(H))
def chunk(t,d):
    c=t+d; return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',W,H,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
open('/tmp/svc_smoke_test.png','wb').write(png)
PYEOF
}

check_up() {
  if ! curl -sf --max-time 3 "localhost:$1/health" >/dev/null 2>&1; then
    echo "❌ 端口 $1 服务不可用, 先启动: svc.sh start $2"; exit 1
  fi
}

test_planner() {
  check_up 8001 planner
  echo "===== planner (8001) 文本 ====="
  python3 - <<'PYEOF'
import json, urllib.request, time
t0=time.time()
body=json.dumps({"model":"qwen3-4b-instruct-2507",
  "messages":[{"role":"user","content":"你是 GUI 自动化 agent 的 planner。目标:打开浏览器查今天天气。输出3步计划,每步一行。"}],
  "max_tokens":150,"temperature":0.7}).encode()
req=urllib.request.Request("http://localhost:8001/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=120))
c=r["choices"][0]
text=c.get("text") or c[next(k for k in c if k not in("index","finish_reason","logprobs"))]["content"]
print(text.strip())
print(f"✅ 通过 ({time.time()-t0:.1f}s, {r['usage']['completion_tokens']} tokens)")
PYEOF
}

test_vision() {
  check_up 8002 vision
  echo "===== vision (8002) 图文理解 ====="
  gen_image
  python3 - <<'PYEOF'
import base64, json, urllib.request, time
b64=base64.b64encode(open('/tmp/svc_smoke_test.png','rb').read()).decode()
t0=time.time()
body=json.dumps({"model":"qwen3-vl-32b-instruct-fp8",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"图中有哪些颜色的色块?分别在什么位置?简短回答。"},
    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}],
  "max_tokens":200,"temperature":0.2}).encode()
req=urllib.request.Request("http://localhost:8002/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=180))
c=r["choices"][0]
text=c.get("text") or c[next(k for k in c if k not in("index","finish_reason","logprobs"))]["content"]
print(text.strip())
ok=any(w in text for w in ("红","绿","蓝"))
print(("✅ 通过" if ok else "⚠️ 回答中未提到预期颜色, 请人工核对")+f" ({time.time()-t0:.1f}s)")
PYEOF
}

test_grounder() {
  check_up 8003 grounder
  echo "===== grounder (8003) 坐标定位 ====="
  gen_image
  python3 - <<'PYEOF'
import base64, json, urllib.request, time
b64=base64.b64encode(open('/tmp/svc_smoke_test.png','rb').read()).decode()
prompt="""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
点击绿色色块的中心"""
t0=time.time()
body=json.dumps({"model":"gta1-7b",
  "messages":[{"role":"user","content":[
    {"type":"text","text":prompt},
    {"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}],
  "max_tokens":200,"temperature":0.1}).encode()
req=urllib.request.Request("http://localhost:8003/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=120))
c=r["choices"][0]
text=c.get("text") or c[next(k for k in c if k not in("index","finish_reason","logprobs"))]["content"]
print(text.strip())
# 绿色块中心真值 (320,240), 块范围 x[213,427] y[160,320]
import re
m=re.search(r"\((\d+)\s*,\s*(\d+)\)", text)
if m:
    x,y=int(m.group(1)),int(m.group(2))
    inside = 213<=x<=427 and 160<=y<=320
    print(f"坐标 ({x},{y}) {'✅ 落在绿色块内, 通过' if inside else '⚠️ 不在绿色块内(真值320,240), 请人工核对'}"+f" ({time.time()-t0:.1f}s)")
else:
    print("⚠️ 输出中没有坐标, 请人工核对")
PYEOF
}

case "$target" in
  planner)  test_planner ;;
  vision)   test_vision ;;
  grounder) test_grounder ;;
  all)      test_planner; echo; test_vision; echo; test_grounder ;;
  *) echo "用法: $0 [vision|grounder|planner|all]"; exit 1 ;;
esac
