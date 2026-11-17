#!/usr/bin/env python3
"""DGX Spark 实时监控面板: GPU / 内存 / CPU / 网络带宽
用法: monitor.py [刷新间隔秒, 默认1]   Ctrl+C 退出
"""
import os, sys, time, shutil, subprocess
from datetime import datetime

INTERVAL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

# ---------- 数据源 ----------

def read_cpu():
    """返回 {core名: (busy, total)}"""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            parts = line.split()
            vals = list(map(int, parts[1:8]))
            idle = vals[3] + vals[4]          # idle + iowait
            out[parts[0]] = (sum(vals) - idle, sum(vals))
    return out

def read_mem():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])       # kB
    return info

def read_net():
    out = {}
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, rest = line.split(":", 1)
            name = name.strip()
            if name.startswith(("lo", "docker", "veth", "br-")):
                continue
            fields = rest.split()
            out[name] = (int(fields[0]), int(fields[8]))  # rx, tx bytes
    return out

def read_gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = [v.strip().strip("[]") for v in r.stdout.strip().split(",")]
        f = lambda v: None if v in ("N/A", "") else float(v)
        return {"util": f(vals[0]), "temp": f(vals[1]), "power": f(vals[2]), "plimit": f(vals[3])}
    except Exception:
        return None

# ---------- 渲染 ----------

def bar(pct, width=28):
    pct = max(0.0, min(100.0, pct))
    n = int(width * pct / 100)
    color = "\033[32m" if pct < 60 else ("\033[33m" if pct < 85 else "\033[31m")
    return f"{color}{'█' * n}{'░' * (width - n)}\033[0m {pct:5.1f}%"

def fmt_rate(bps):  # bytes/s
    if bps >= 1 << 20: return f"{bps / (1 << 20):8.2f} MB/s"
    if bps >= 1 << 10: return f"{bps / (1 << 10):8.1f} KB/s"
    return f"{bps:8.0f} B/s "

prev_cpu, prev_net = read_cpu(), read_net()
t_ref = time.time()
time.sleep(0.2)

try:
    while True:
        t0 = time.time()
        cpu, net, mem, gpu = read_cpu(), read_net(), read_mem(), read_gpu()
        cols = shutil.get_terminal_size((100, 30)).columns

        out = ["\033[H\033[J"]  # 清屏
        out.append(f"\033[1m DGX Spark 监控 \033[0m  {datetime.now():%Y-%m-%d %H:%M:%S}   (Ctrl+C 退出, 刷新 {INTERVAL}s)")
        out.append("─" * min(cols, 72))

        # GPU (GB10 统一内存, 显存占用看下面的内存条)
        if gpu:
            util = gpu["util"] or 0.0
            p = f"{gpu['power']:.0f}W" if gpu["power"] else "N/A"
            pl = f"/{gpu['plimit']:.0f}W" if gpu["plimit"] else ""
            t = f"{gpu['temp']:.0f}°C" if gpu["temp"] else "N/A"
            out.append(f" GPU (GB10)  {bar(util)}   {t}  {p}{pl}")
        else:
            out.append(" GPU         nvidia-smi 不可用")

        # 内存 (统一内存 = CPU+GPU 共享)
        tot = mem["MemTotal"]; avail = mem["MemAvailable"]
        used_pct = (tot - avail) / tot * 100
        out.append(f" 内存        {bar(used_pct)}   已用 {(tot-avail)/2**20:.1f} / {tot/2**20:.0f} GiB, 可用 {avail/2**20:.1f} GiB")

        # CPU 总览 + 每核
        busy_d = cpu["cpu"][0] - prev_cpu["cpu"][0]
        tot_d = cpu["cpu"][1] - prev_cpu["cpu"][1]
        cpu_pct = busy_d / tot_d * 100 if tot_d else 0.0
        out.append(f" CPU (20核)  {bar(cpu_pct)}")
        cores = sorted((k for k in cpu if k != "cpu"), key=lambda x: int(x[3:]))
        line = "      "
        for c in cores:
            d = cpu[c][0] - prev_cpu[c][0]; t_ = cpu[c][1] - prev_cpu[c][1]
            p = d / t_ * 100 if t_ else 0.0
            n = int(4 * p / 100)
            color = "\033[32m" if p < 60 else ("\033[33m" if p < 85 else "\033[31m")
            line += f"{c[3:]:>3}{color}{'▮' * n}{'▯' * (4 - n)}\033[0m "
            if (int(c[3:]) + 1) % 5 == 0:
                out.append(line); line = "      "
        if line.strip(): out.append(line)

        # 网络
        out.append(" 网络")
        any_iface = False
        for name, (rx, tx) in sorted(net.items()):
            if name not in prev_net:
                continue
            rxs = (rx - prev_net[name][0]) / max(t0 - t_ref, 1e-6)
            txs = (tx - prev_net[name][1]) / max(t0 - t_ref, 1e-6)
            any_iface = True
            out.append(f"      {name:<10} ↓{fmt_rate(rxs)}  ↑{fmt_rate(txs)}")
        if not any_iface:
            out.append("      (无活动接口)")

        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()

        prev_cpu, prev_net, t_ref = cpu, net, t0
        time.sleep(max(0.1, INTERVAL - (time.time() - t0)))
except KeyboardInterrupt:
    print("\n已退出")
