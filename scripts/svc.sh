#!/usr/bin/env bash
# 模型服务管理脚本
# 用法: svc.sh <start|stop|restart|status|logs> [vision|grounder|planner|all]
#   start   启动(按 先大后小 顺序, 并等待健康检查通过)
#   stop    停止
#   restart 重建容器(改完 docker-compose.yml 后用这个)
#   status  容器状态 + 端口健康 + 内存
#   logs    跟踪某个服务的日志
set -euo pipefail

COMPOSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../deploy" && pwd)"
ORDER=(vision grounder planner)   # 启动顺序: 先大后小, 避免显存测量竞争

# docker 权限自检: 无权限则借 sg 重新执行本脚本
if ! docker info >/dev/null 2>&1; then
  exec sg docker -c "$0 $*"
fi

cmd="${1:-}"; target="${2:-all}"
usage() { echo "用法: $0 <start|stop|restart|status|logs> [vision|grounder|planner|all]"; exit 1; }
[[ -z "$cmd" ]] && usage

port_of() {
  case "$1" in
    planner)  echo 8001 ;;
    vision)   echo 8002 ;;
    grounder) echo 8003 ;;
  esac
}

targets() {
  if [[ "$target" == "all" ]]; then echo "${ORDER[@]}"; else echo "$target"; fi
}

wait_ready() {
  local svc="$1" port; port=$(port_of "$svc")
  echo -n "等待 $svc 就绪 (端口 $port)"
  for _ in $(seq 1 60); do
    if curl -sf --max-time 3 "localhost:$port/health" >/dev/null 2>&1; then
      echo " ✅"; return 0
    fi
    echo -n "."; sleep 10
  done
  echo " ❌ 超时, 查日志: $0 logs $svc"; return 1
}

case "$cmd" in
  start)
    for s in $(targets); do
      (cd "$COMPOSE_DIR" && docker compose up -d "$s")
      wait_ready "$s"
    done
    ;;
  stop)
    if [[ "$target" == "all" ]]; then
      (cd "$COMPOSE_DIR" && docker compose stop)
    else
      (cd "$COMPOSE_DIR" && docker compose stop "$target")
    fi
    ;;
  restart)
    for s in $(targets); do
      (cd "$COMPOSE_DIR" && docker compose up -d --force-recreate "$s")
      wait_ready "$s"
    done
    ;;
  status)
    (cd "$COMPOSE_DIR" && docker compose ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}')
    echo
    for s in planner vision grounder; do
      p=$(port_of "$s")
      if curl -sf --max-time 3 "localhost:$p/health" >/dev/null 2>&1; then
        echo "$s (:$p) ✅ healthy"
      else
        echo "$s (:$p) ❌ 不可用"
      fi
    done
    echo
    free -h | sed -n '1,2p'
    ;;
  logs)
    [[ "$target" == "all" ]] && usage
    docker logs -f "vllm-$target"
    ;;
  *)
    usage
    ;;
esac
