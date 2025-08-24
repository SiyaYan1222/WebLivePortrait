set -euo pipefail

NAME=${NAME:-web_liveportrait}
IMAGE=${IMAGE:-shaoguo/faster_liveportrait:v3}
HOST_PROJ=${HOST_PROJ:-"$(pwd -P)"}
DISPLAY_VAR=${DISPLAY:-:0}
PORT=${PORT:-9870}

# Camera devices (adjust if needed)
WEBCAM0=${WEBCAM0:-/dev/video0}
WEBCAM1=${WEBCAM1:-/dev/video1}

usage() {
  echo "Usage: $0 {start|webcam [index]|shell|logs|stop|rm}"
  echo "  start           Start container (detached)"
  echo "  webui           Run container with web UI"
  echo "  webcam [index] [source]  Run webcam with camera index and source image"
  echo "  shell           Enter a bash shell in the container"
  echo "  logs            Follow container logs"
  echo "  stop            Stop container (also revoke X access)"
  echo "  rm/remove       Remove container (also revoke X access)"
}

start_container() {
  command -v docker >/dev/null || { echo "ERROR: docker not found."; exit 1; }
  [ -d "$HOST_PROJ" ] || { echo "ERROR: Project dir not found: $HOST_PROJ"; exit 1; }

  # Allow X11 access for local root (temporary; revoked on stop/rm)
  xhost +local:root >/dev/null 2>&1 || true

  # Build run options
  RUN_OPTS=( --gpus=all --name "$NAME" --ipc=host --shm-size=8g -d )
  [ -e "$WEBCAM0" ] && RUN_OPTS+=( --device "$WEBCAM0" )
  [ -e "$WEBCAM1" ] && RUN_OPTS+=( --device "$WEBCAM1" )
  # Nvidia device nodes (not strictly required with --gpus=all, but kept for parity)
  for d in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [ -e "$d" ] && RUN_OPTS+=( --device "$d" )
  done
  RUN_OPTS+=(
    -e DISPLAY="$DISPLAY_VAR" --network host
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro
    -v /usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:ro
    -v "$HOST_PROJ":/root/WebLivePortrait
  )

  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Container '$NAME' already exists; starting…"
    docker start "$NAME" >/dev/null
  else
    echo "Creating & starting '$NAME' from '$IMAGE'…"
    docker run "${RUN_OPTS[@]}" "$IMAGE"  \
    bash -lic 'cd /root/WebLivePortrait && \
               pip install -r requirements.txt && \
               echo "[Inside container] Dependencies installed, keeping container alive..." && \
               sleep infinity'
  fi

  echo "Container '$NAME' is up. Use: $0 webcam 0"
}

run_webcam() {
  local idx="${1:-0}"                                   # camera index
  local src="${2:-assets/examples/source/test.png}"     # source image path
  docker exec -it "$NAME" bash -lic "cd /root/WebLivePortrait && \
    python run.py \
      --src_image \"${src}\" \
      --dri_video ${idx} \
      --cfg configs/trt_infer.yaml \
      --realtime"
}

run_webui(){
  CMD='cd /root/WebLivePortrait && python webui.py --mode trt'
  docker exec -it "$NAME" bash -lic "$CMD"
  echo "Web UI should be reachable on http://localhost:${PORT}"
}

enter_shell() {
  docker exec -it "$NAME" bash
}

follow_logs() {
  docker logs -f "$NAME"
}

stop_container() {
  docker stop "$NAME"
  xhost -local:root >/dev/null 2>&1 || true
}

remove_container() {
  docker rm -f "$NAME"
  xhost -local:root >/dev/null 2>&1 || true
}

cmd="${1:-}"
case "$cmd" in
  start)  start_container ;;
  webui)  run_webui ;;
  webcam) shift; run_webcam "${1:-0}" "${2:-assets/examples/source/test.png}" ;;
  shell)  enter_shell ;;
  logs)   follow_logs ;;
  stop)   stop_container ;;
  rm)     remove_container ;;
  remove) remove_container ;;
  *)      usage; exit 1 ;;
esac
