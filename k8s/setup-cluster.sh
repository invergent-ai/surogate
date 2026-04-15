#!/bin/bash

# This script sets up the Surogate Kubernetes cluster
# apt packages required: certutil jq libnss3-tools wget envsubst

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export K3D_IMAGE="ghcr.io/invergent-ai/k3s-v1.35.3-k3s1-cuda-12.9.1-cudnn-runtime-ubuntu24.04"
export SUROGATE_DIR="${HOME}/.surogate"
export LAKEFS_DIR="${SUROGATE_DIR}/lakefs"
export GARAGE_DIR="${SUROGATE_DIR}/garage"
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

if [ -d "$SUROGATE_DIR/lakefs" ]; then
    echo -e "${YELLOW}✖ WARN: Runtime directory '${SUROGATE_DIR}/lakefs' already exists.${NC}"
fi
if [ -d "$GARAGE_DIR" ]; then
    echo -e "${YELLOW}✖ WARN: Runtime directory '${GARAGE_DIR}' already exists.${NC}"
fi

mkdir -p "${SUROGATE_DIR}/bin"
mkdir -p "${LAKEFS_DIR}"
mkdir -p "${GARAGE_DIR}/meta"
mkdir -p "${GARAGE_DIR}/data"
sudo chmod -R 777 "${GARAGE_DIR}"

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
    "$HELM" repo add bitnami https://charts.bitnami.com/bitnami
    "$HELM" repo add charts-derwitt-dev https://charts.derwitt.dev
    "$HELM" repo update
}

create_cluster() {
    for host in k8s.localhost studio.k8s.localhost lakefs.k8s.localhost lakefs-s3.k8s.localhost metrics.k8s.localhost surogates.k8s.localhost garage.k8s.localhost; do
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
    "$HELM" install traefik traefik/traefik --version 35.4.0 -n kube-system -f "$SCRIPT_DIR/traefik/values.yml" > /dev/null
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
    "$HELM" install lakefs lakefs/lakefs -n lakefs -f - <<< "$rendered_values" > /dev/null
    "$KUBECTL" apply -f "${SCRIPT_DIR}/lakefs/s3-service.yml"
}

install_gpu() {
    "$KUBECTL" create namespace nvidia-gpu-operator
    "$KUBECTL" apply -f "${SCRIPT_DIR}/gpu/configmap.yml"
    "$HELM" install nvidia-gpu-operator nvidia/gpu-operator --version=v26.3.0 -n nvidia-gpu-operator -f "${SCRIPT_DIR}/gpu/values.yml" > /dev/null
}

install_metrics() {
    "$KUBECTL" create namespace monitoring
    "$HELM" install kube-prometheus-stack prometheus-community/kube-prometheus-stack -f "${SCRIPT_DIR}/metrics/values.yml" -n monitoring > /dev/null
    "$KUBECTL" apply -f "${SCRIPT_DIR}/metrics/ingress.yml"
    "$KUBECTL" apply -f "${SCRIPT_DIR}/metrics/gpu_scraper.yml"
}

install_role() {
    "$KUBECTL" apply -f "${SCRIPT_DIR}/role/cluster-role.yml"
}

install_db() {
    "$HELM" install surogate-db bitnami/postgresql -f "${SCRIPT_DIR}/db/values.yml" > /dev/null
    # if needed, you can retrieve the generated password for the 'postgres' user with:
    # export POSTGRES_PASSWORD=$("$KUBECTL" get secret --namespace surogate-db surogate-db-postgresql -o jsonpath="{.data.password}" | base64 -d)
}

install_redis() {
    "$HELM" install surogates-redis bitnami/redis -f "${SCRIPT_DIR}/redis/values.yml" > /dev/null
}

install_garage() {
    "$HELM" install surogates-s3 charts-derwitt-dev/garage --version 2.3.1 -f "${SCRIPT_DIR}/garage/values.yml" > /dev/null
    "$KUBECTL" apply -f "${SCRIPT_DIR}/garage/ingress.yml"
}

configure_garage() {
    local pod node_id output
    "$KUBECTL" rollout status daemonset/surogates-s3-garage --timeout=120s > /dev/null

    pod=$("$KUBECTL" get pod -l app.kubernetes.io/instance=surogates-s3 -o jsonpath='{.items[0].metadata.name}')
    node_id=$("$KUBECTL" exec "$pod" -- /garage status 2>/dev/null | awk 'NR>2 && $1 ~ /^[0-9a-f]+$/ {print $1; exit}')

    if [ -z "$node_id" ]; then
        echo -e "${RED}✖ ERROR: Could not determine Garage node id.${NC}"
        return 1
    fi

    "$KUBECTL" exec "$pod" -- /garage layout assign -z dc1 -c 1G "$node_id" > /dev/null 2>&1 || true
    "$KUBECTL" exec "$pod" -- /garage layout apply --version 1 > /dev/null 2>&1 || true
    "$KUBECTL" exec "$pod" -- /garage bucket create surogates > /dev/null 2>&1 || true

    if ! "$KUBECTL" exec "$pod" -- /garage key info surogates-key --show-secret > /dev/null 2>&1; then
        "$KUBECTL" exec "$pod" -- /garage key create surogates-key > /dev/null
    fi
    output=$("$KUBECTL" exec "$pod" -- /garage key info surogates-key --show-secret)
    GARAGE_ACCESS_KEY_ID=$(echo "$output" | awk -F': ' '/Key ID/ {print $2}' | tr -d ' ')
    GARAGE_SECRET_ACCESS_KEY=$(echo "$output" | awk -F': ' '/Secret key/ {print $2}' | tr -d ' ')

    "$KUBECTL" exec "$pod" -- /garage bucket allow --read --write --owner surogates --key surogates-key > /dev/null 2>&1 || true
    # Agent sessions create a per-session bucket at runtime, so the
    # key needs the ``create-bucket`` permission at the key level.
    "$KUBECTL" exec "$pod" -- /garage key allow surogates-key --create-bucket > /dev/null 2>&1 || true
}

install_surogates() {
    "$KUBECTL" create namespace surogates
    if [ -d "${SUROGATES_CHART_DIR:-/work/surogates/helm/surogates}" ]; then
        "$HELM" upgrade --install surogates "${SUROGATES_CHART_DIR:-/work/surogates/helm/surogates}" \
            --namespace surogates -f "${SCRIPT_DIR}/surogates/values.yml" > /dev/null
        "$KUBECTL" apply -f "${SCRIPT_DIR}/surogates/ingress.yml"
    else
        echo -e "${YELLOW}✖ WARN: Surogates Helm chart not found; skipping surogates release (namespace created).${NC}"
    fi
}

create_server_config() {
    cat >"${SUROGATE_DIR}/config.yaml" <<EOF
host: 0.0.0.0
port: 8888
database_url: postgresql+asyncpg://surogate:surogate@127.0.0.1:32432/surogate
dstack_database_url: postgresql+asyncpg://dstack:dstack@127.0.0.1:32432/dstack
surogates_database_url: postgresql+asyncpg://surogates:surogates@127.0.0.1:32432/surogates
lakefs_endpoint: https://lakefs.k8s.localhost
lakefs_s3_endpoint: https://lakefs-s3.k8s.localhost
lakefs_access_key: $LAKEFS_ACCESS_KEY_ID
lakefs_secret_key: $LAKEFS_SECRET_ACCESS_KEY
agent_s3_access_key: $GARAGE_ACCESS_KEY_ID
agent_s3_secret_key: $GARAGE_SECRET_ACCESS_KEY
EOF
}

# Check if any k3d clusters exist
existing=$("$K3D" cluster list -o json | jq -r '.[].name')
if [ -n "$existing" ]; then
    echo -e "${RED}✖ ERROR: Existing k3d cl6usters found, please delete all clusters and try again.${NC}"
    exit 1
fi

setup_helm_repositories
setup_lakefs
create_cluster

sleep 3 # wait for cluster to be ready

"$K3D" kubeconfig write "$CLUSTER_NAME" --output "$SUROGATE_DIR/kubeconfig"
echo -e "${CYAN}  Run: export KUBECONFIG=$SUROGATE_DIR/kubeconfig${NC}"

export KUBECONFIG="$SUROGATE_DIR/kubeconfig"

install_traefik
install_db
install_redis
install_garage
configure_garage
install_gpu
install_lakefs
install_metrics
install_role
install_surogates
create_server_config

echo -e "${GREEN}✓ Cluster setup complete!${NC}"