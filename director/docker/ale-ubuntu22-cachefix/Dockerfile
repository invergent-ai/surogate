FROM agentslastexam/ale-ubuntu22-docker@sha256:78ec11afeb0008ed8bc2b59cf9c90c05e63d1ac66b9d3e7cb0fada10695fca6f

USER root
RUN install -d -o user -g user -m 0755 /home/user/.cache \
    && chown -R user:user /home/user/.cache
USER user
