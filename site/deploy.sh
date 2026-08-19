#!/usr/bin/env bash
# Deploy the minimlx.com landing page to the k3s cluster.
# Run on the cluster host (uses `sudo k3s kubectl`), or set KUBECTL=kubectl
# with a KUBECONFIG pointing at it.
set -euo pipefail
cd "$(dirname "$0")"

KUBECTL=${KUBECTL:-"sudo k3s kubectl"}

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp index.html "$STAGE/"
cp ../assets/preview.png "$STAGE/"

if command -v sha256sum >/dev/null; then
  HASH=$(cat "$STAGE"/index.html "$STAGE"/preview.png | sha256sum | cut -c1-16)
else
  HASH=$(cat "$STAGE"/index.html "$STAGE"/preview.png | shasum -a 256 | cut -c1-16)
fi

$KUBECTL apply -f k8s.yaml
$KUBECTL create configmap minimlx-site \
  --namespace minimlx \
  --from-file="$STAGE/index.html" \
  --from-file="$STAGE/preview.png" \
  --dry-run=client -o yaml | $KUBECTL apply -f -

$KUBECTL -n minimlx patch deployment minimlx-web --type=merge \
  -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"minimlx.com/content-hash\":\"$HASH\"}}}}}"

$KUBECTL -n minimlx rollout status deployment/minimlx-web --timeout=180s
echo "deployed: content $HASH"
