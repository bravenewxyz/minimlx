#!/usr/bin/env bash
# Deploy the minimlx.com landing page to the k3s cluster.
# Usage: ./deploy.sh          (run on the cluster host, or with KUBECONFIG set)
set -euo pipefail
cd "$(dirname "$0")"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp index.html "$STAGE/"
cp ../assets/preview.png "$STAGE/"

HASH=$(cat "$STAGE"/index.html "$STAGE"/preview.png | shasum -a 256 | cut -c1-16)

kubectl apply -f k8s.yaml
kubectl create configmap minimlx-site \
  --namespace minimlx \
  --from-file="$STAGE/index.html" \
  --from-file="$STAGE/preview.png" \
  --dry-run=client -o yaml | kubectl apply -f -

# A ConfigMap change does not restart pods on its own; the annotation does.
kubectl -n minimlx patch deployment minimlx-site --type=merge \
  -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"minimlx.com/content-hash\":\"$HASH\"}}}}}"

kubectl -n minimlx rollout status deployment/minimlx-site --timeout=120s
echo "deployed: content $HASH"
