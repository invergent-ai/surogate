#!/bin/bash

# This script sets up the Surogate Kubernetes cluster
# apt packages required: certutil jq libnss3-tools wget envsubst

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export K3D_IMAGE="ghcr.io/invergent-ai/k3s-v1.35.3-k3s1-cuda-12.9.1-cudnn-runtime-ubuntu24.04"
export SUROGATE_DIR="${HOME}/.surogate"
export LAKEFS_DIR="${SUROGATE_DIR}/lakefs"
export HF_CACHE="${HOME}/.cache/huggingface"
export CLUSTER_NAME="surogate"
export SERVERS=1
export AGENTS=1
export API_PORT=6443
export HTTP_PORT=80
export HTTPS_PORT=443
export LAKEFS_SECRET_KEY="ipsKBPkU3D1pdrWXvDQHowVdX7m9bK0s"

export KUBECTL="${SUROGATE_DIR}/bin/kubectl"
export HELM="${SUROGATE_DIR}/bin/helm"
export MKCERT="${SUROGATE_DIR}/bin/mkcert"
export K3D="${SUROGATE_DIR}/bin/k3d"
export LAKECTL="${SUROGATE_DIR}/bin/lakectl"
export LAKEFS="${SUROGATE_DIR}/bin/lakefs"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

 if [ -d "$SUROGATE_DIR" ]; then
    if [ -d "$SUROGATE_DIR/lakefs" ]; then
        echo -e "${YELLOW}✖ WARN: Runtime directory '${SUROGATE_DIR}/lakefs' already exists.${NC}"
    fi
else
    mkdir -p "${SUROGATE_DIR}/bin"
    mkdir -p "${LAKEFS_DIR}"
fi

# Install kubectl
if [ ! -f "${SUROGATE_DIR}/bin/kubectl" ]; then
    kubectl_version=$(curl -L -s https://dl.k8s.io/release/stable.txt)
    curl -L -o "${SUROGATE_DIR}/bin/kubectl" "https://dl.k8s.io/release/${kubectl_version}/bin/linux/amd64/kubectl"
    chmod +x "${SUROGATE_DIR}/bin/kubectl"
fi

# Install helm
if [ ! -f "${SUROGATE_DIR}/bin/helm" ]; then
    mkdir -p "${SUROGATE_DIR}/tmp"
    wget -q -O "${SUROGATE_DIR}/tmp/helm-v4.1.3-linux-amd64.tar.gz" https://get.helm.sh/helm-v4.1.3-linux-amd64.tar.gz
    cd "${SUROGATE_DIR}/tmp"
    tar -xzf helm-v4.1.3-linux-amd64.tar.gz
    mv linux-amd64/helm "${SUROGATE_DIR}/bin/helm"
    chmod +x "${SUROGATE_DIR}/bin/helm"
    rm -rf "${SUROGATE_DIR}/tmp"
fi

# Install k3d
if [ ! -f "${SUROGATE_DIR}/bin/k3d" ]; then
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | K3D_INSTALL_DIR="${SUROGATE_DIR}/bin" bash
fi

# Install mkcert
if [ ! -f "${SUROGATE_DIR}/bin/mkcert" ]; then
    wget -q -O "${SUROGATE_DIR}/bin/mkcert" https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64
    chmod +x "${SUROGATE_DIR}/bin/mkcert"
fi

# Install lakeFS CLIs
if [ ! -f "${SUROGATE_DIR}/bin/lakectl" ] || [ ! -f "${SUROGATE_DIR}/bin/lakefs" ]; then
    curl -o "${SUROGATE_DIR}/bin/lakectl" https://densemax.s3.eu-central-1.amazonaws.com/lakectl
    curl -o "${SUROGATE_DIR}/bin/lakefs" https://densemax.s3.eu-central-1.amazonaws.com/lakefs
    chmod +x "${SUROGATE_DIR}/bin/lakectl" "${SUROGATE_DIR}/bin/lakefs"
fi

setup_helm_repositories() {
    "$HELM" repo add traefik https://traefik.github.io/charts
    "$HELM" repo add lakefs https://charts.lakefs.io
    "$HELM" repo add prometheus-community https://prometheus-community.github.io/helm-charts
    "$HELM" repo add nvidia https://helm.ngc.nvidia.com/nvidia
    "$HELM" repo add dcgm-exporter https://nvidia.github.io/dcgm-exporter/helm-charts
    "$HELM" repo update
}

create_cluster() {
    for host in k8s.localhost studio.k8s.localhost lakefs.k8s.localhost lakefs-s3.k8s.localhost metrics.k8s.localhost; do
        grep -qF "$host" /etc/hosts || sudo sh -c "echo '127.0.0.1 $host' >> /etc/hosts"
    done
    
    tmp_config=$(mktemp /tmp/k3d-config-XXXXXX.yaml)
    envsubst < "${SCRIPT_DIR}/cluster.yml" > "$tmp_config"
    "$K3D" cluster create --config "$tmp_config" --gpus all
    rm -f "$tmp_config"
}

install_traefik() {
    "$MKCERT" -key-file "${SUROGATE_DIR}/ssl.key.pem" -cert-file "${SUROGATE_DIR}/ssl.cert.pem" "*.k8.localhost"
    "$KUBECTL" create secret generic traefik-tls-secret --from-file=tls.crt="${SUROGATE_DIR}/ssl.cert.pem" --from-file=tls.key="${SUROGATE_DIR}/ssl.key.pem" -n kube-system
    "$HELM" install traefik traefik/traefik --version 35.4.0 -n kube-system -f "$SCRIPT_DIR/traefik/values.yml"
    "$KUBECTL" apply -f "${SCRIPT_DIR}/traefik/middleware.yml"
}

setup_lakefs() {
    local output
    tmp_config=$(mktemp /tmp/lakefs-config-XXXXXX.yaml)
    cat >"${tmp_config}" <<EOF
auth:
  encrypt:
    secret_key: $LAKEFS_SECRET_KEY
database:
  type: local
  local:
    path: ${LAKEFS_DIR}/db
    sync_writes: true
blockstore:
  type: local
  signing:
    secret_key: $LAKEFS_SECRET_KEY
  local:
    path: ${LAKEFS_DIR}/data
committed:
  local_cache:
    dir: ${LAKEFS_DIR}/cache
EOF

    output=$("$LAKEFS" -c "${tmp_config}" setup --user-name admin 2>/dev/null)
    LAKEFS_ACCESS_KEY_ID=$(echo "$output" | grep 'access_key_id:' | awk '{print $2}')
    LAKEFS_SECRET_ACCESS_KEY=$(echo "$output" | grep 'secret_access_key:' | awk '{print $2}')
    rm -f "$tmp_config"

    cat >"${SUROGATE_DIR}/lakectl.yaml" <<EOF
credentials:
    access_key_id: $LAKEFS_ACCESS_KEY_ID
    secret_access_key: $LAKEFS_SECRET_ACCESS_KEY
experimental:
    local:
        posix_permissions:
            enabled: false
local:
    skip_non_regular_files: false
server:
    endpoint_url: https://lakefs.k8s.localhost/api/v1
    retries:
        enabled: true
        max_attempts: 4
        max_wait_interval: 30s
        min_wait_interval: 200ms
EOF
   
    chmod -R 777 "${LAKEFS_DIR}"
}

install_lakefs() {
    "$KUBECTL" create namespace lakefs
    "$KUBECTL" apply -f "${SCRIPT_DIR}/lakefs/volume.yml"

    local rendered_values
    rendered_values=$(envsubst < "${SCRIPT_DIR}/lakefs/values.yml")
    "$HELM" install lakefs lakefs/lakefs -n lakefs -f - <<< "$rendered_values"
    "$KUBECTL" apply -f "${SCRIPT_DIR}/lakefs/s3-service.yml"
}

install_gpu() {
    "$KUBECTL" create namespace nvidia-gpu-operator
    "$KUBECTL" apply -f "${SCRIPT_DIR}/gpu/configmap.yml"
    "$HELM" install nvidia-gpu-operator nvidia/gpu-operator --version=v26.3.0 --wait -n nvidia-gpu-operator -f "${SCRIPT_DIR}/gpu/values.yml"
}

install_metrics() {
    "$KUBECTL" create namespace monitoring
    "$HELM" install kube-prometheus-stack prometheus-community/kube-prometheus-stack -f "${SCRIPT_DIR}/metrics/values.yml" -n monitoring
    "$KUBECTL" apply -f "${SCRIPT_DIR}/metrics/ingress.yml"
    "$KUBECTL" apply -f "${SCRIPT_DIR}/metrics/gpu_scraper.yml"
}

create_server_config() {
    cat >"${SUROGATE_DIR}/config.yaml" <<EOF
host: 127.0.0.1
port: 8888
database_url: sqlite+aiosqlite:///${SUROGATE_DIR}/surogate.db
lakefs_endpoint: https://lakefs.k8s.localhost
lakefs_s3_endpoint: https://lakefs-s3.k8s.localhost
lakefs_access_key: $LAKEFS_ACCESS_KEY_ID
lakefs_secret_key: $LAKEFS_SECRET_ACCESS_KEY
EOF
}

# Check if any k3d clusters exist
existing=$("$K3D" cluster list -o json | jq -r '.[].name')
if [ -n "$existing" ]; then
    echo -e "${RED}✖ ERROR: Existing k3d clusters found, please delete all clusters and try again.${NC}"
    exit 1
fi

setup_helm_repositories
setup_lakefs
create_cluster

sleep 3 # wait for cluster to be ready

"$K3D" kubeconfig write "$CLUSTER_NAME" --output "$SUROGATE_DIR/kubeconfig"
echo -e "${CYAN}  Run: export KUBECONFIG=$SUROGATE_DIR/kubeconfig${NC}"

export KUBECONFIG="$SUROGATE_DIR/kubeconfig"

install_gpu
install_traefik
install_lakefs
install_metrics
create_server_config

echo -e "${GREEN}✓ Cluster setup complete!${NC}"