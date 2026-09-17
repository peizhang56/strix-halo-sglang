#!/usr/bin/env bash
# Launch the SGLang gfx1151 dev container (detached, idles on `sleep infinity`).
# Run on the HOST; then work inside via `docker exec -it <name> bash`.
#
# Only libdxcore.so and librocdxg.so are mounted from the host: there is no /dev/kfd
# under WSL, so the container's HSA runtime dlopens librocdxg.so to reach /dev/dxg.
# ROCm userspace comes from the image. Do NOT also mount the host libhsa-runtime64 --
# it is byte-identical to the image's copy. See kb/rocm-wsl-mounts.md.
set -euo pipefail

NAME=sglang-dev
IMAGE=rocm/sgl-dev:v0.5.19-rocm724-gfx1151-20260914
HF_CACHE="$HOME/.cache/huggingface"
HOST_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# Where HOST_DIR shows up inside the container. Nothing in the repo's scripts
# depends on this value -- they all resolve paths relative to themselves -- so it
# is a plain default, not a contract.
MOUNT_DIR="${SGLANG_MOUNT_DIR:-/workspace}"
PORT=30000
RECREATE=0

usage() {
  cat <<EOF
Usage: ./launch_docker.sh [options]
  --name NAME        container name   (default: $NAME)
  --image IMAGE      docker image     (default: $IMAGE)
  --hf-cache PATH    HF model cache   (default: \$HOME/.cache/huggingface)
  --host-dir PATH    host dir to mount (default: git root of cwd)
  --mount-dir PATH   where it appears in the container (default: $MOUNT_DIR)
  --port PORT        published port   (default: $PORT)
  --recreate         remove an existing container of the same name first
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --name)     NAME=$2; shift 2 ;;
    --image)    IMAGE=$2; shift 2 ;;
    --hf-cache) HF_CACHE=$2; shift 2 ;;
    --host-dir) HOST_DIR=$2; shift 2 ;;
    --mount-dir) MOUNT_DIR=$2; shift 2 ;;
    --port)     PORT=$2; shift 2 ;;
    --recreate) RECREATE=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -f /.dockerenv ] && { echo "Already inside a container; run this on the host." >&2; exit 1; }

if [ "$(docker ps -aq -f "name=^${NAME}$")" ]; then
  if [ "$RECREATE" = 1 ]; then
    docker rm -f "$NAME" >/dev/null
  else
    docker start "$NAME" >/dev/null 2>&1 || true
    echo "Container '$NAME' already exists (reused). Use --recreate to rebuild."
    echo "Exec: docker exec -it $NAME bash"
    exit 0
  fi
fi

HOST_DIR=$(readlink -f "$HOST_DIR")
mkdir -p "$HF_CACHE"

MOUNTS=()
add_ro() { [ -e "$1" ] && MOUNTS+=(-v "$1:$2:ro") || { echo "error: missing $1" >&2; exit 1; }; }
add_ro /usr/lib/wsl/lib/libdxcore.so /usr/lib/libdxcore.so
add_ro /opt/rocm/lib/librocdxg.so /usr/lib/librocdxg.so

# Corporate TLS interception: the image ships a stock CA bundle, so HTTPS to
# huggingface.co fails with "unable to get local issuer certificate". Mount the host's
# extra roots and rebuild the bundle in-container (below). SSL_CERT_FILE/REQUESTS_CA_BUNDLE
# point Python at the system bundle, since certifi ships its own and ignores it.
HOST_CA_DIR=/usr/local/share/ca-certificates
[ -d "$HOST_CA_DIR" ] && [ -n "$(ls -A "$HOST_CA_DIR" 2>/dev/null)" ] \
  && MOUNTS+=(-v "$HOST_CA_DIR:/host-ca:ro")

docker run -d --name "$NAME" \
  --device=/dev/dxg \
  "${MOUNTS[@]}" \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16g \
  -p "${PORT}:${PORT}" \
  -e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$HOST_DIR:$MOUNT_DIR" \
  -w "$MOUNT_DIR" \
  "$IMAGE" sleep infinity >/dev/null

if [ -d "$HOST_CA_DIR" ] && [ -n "$(ls -A "$HOST_CA_DIR" 2>/dev/null)" ]; then
  docker exec "$NAME" bash -c \
    'cp /host-ca/*.crt /usr/local/share/ca-certificates/ 2>/dev/null; update-ca-certificates' \
    >/dev/null 2>&1 && echo "Installed host CA roots into container"
fi

echo "Started '$NAME' ($IMAGE)"
echo "  $MOUNT_DIR -> $HOST_DIR"
echo "  hf cache   -> $HF_CACHE"
echo "  port       -> $PORT"
echo "Exec: docker exec -it $NAME bash"
