FROM fugu/ale-ubuntu22-docker:20260721-cachefix-v1

USER root
RUN mv /etc/apt/sources.list.d/google-chrome.list /tmp/google-chrome.list.disabled \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends fuse-overlayfs \
    && rm -rf /var/lib/apt/lists/* \
    && mv /tmp/google-chrome.list.disabled /etc/apt/sources.list.d/google-chrome.list
USER user
