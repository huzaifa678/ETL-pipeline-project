set -euo pipefail

CLUSTER="${1:?usage: kind-load.sh <cluster> <image>}"
IMAGE="${2:?usage: kind-load.sh <cluster> <image>}"

TAR="$(mktemp -t kind-img-XXXXXX.tar)"
trap 'rm -f "$TAR"' EXIT

echo ">> pulling $IMAGE"
docker pull "$IMAGE"
docker save "$IMAGE" -o "$TAR"

for node in $(kind get nodes --name "$CLUSTER"); do
  echo ">> importing into $node"
  docker exec -i "$node" ctr -n k8s.io images import - < "$TAR"
done
echo ">> done: $IMAGE"
