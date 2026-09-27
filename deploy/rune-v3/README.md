# Rune v3 deployment

This bundle serves the local NVFP4 experts / BF16 dense-and-attention artifact with vision,
without speculative decoding. The model and NVFP4 checkpoint remain on this box. It exposes
only decisions, their two aliases, model discovery and health. The hostname and trusted TLS
certificate are supplied on deployment day.

The startup wrapper also supports `RUNE_SPEC=dflash` and `RUNE_DRAFT_TOKENS=7` when
`RUNE_ARTIFACT` points to a local paired target/DFlash artifact. With the decision bypass,
candidate scoring uses the target alone while multi-token reasoning retains speculation.
DFlash mode also enables the measured packed-prefill setting. The default remains
`RUNE_SPEC=none`. See [DFlash readout validation](DFLASH_READOUT.md)
for measurements and numerical limits before selecting a serving configuration.

The initial image budget is **1,120 soft tokens per image**. The engine flag is
`--gemma-image-tokens 1120`; no artifact rewriting or new quantization is required. See the
[measurement report](RESULTS.md) for the accuracy/latency comparison.

## Admission policy

| Limit | Value |
|---|---:|
| Global request rate | 16 requests/s, burst 16 |
| In-flight inference HTTP requests | 16 |
| Per-IP rate at the proxy | 1 request/s, no extra burst (configurable) |
| Per-IP in-flight requests | 8 |
| Concurrent thinking decisions | 2 |
| Concurrent image decisions | 4 |
| Engine pending requests | 8; 2-second admission timeout |
| HTTP request body | 16 MiB |
| Context | 16,384 tokens |
| Scheduler lanes / prefill window | 64 / 8,192 tokens |

The concurrency limit bounds expensive image and thinking traffic even when the rate alone
would admit work faster than that workload can finish. Excess requests get JSON HTTP 429 with
`Retry-After`. Retry with jitter. The native limiter authenticates before consuming capacity;
the edge also limits unauthenticated traffic. The proxy keys clients by the socket peer,
ignoring client-supplied forwarding headers. If a CDN is added, configure its trusted address
ranges before changing client-IP handling.

Set the per-IP policy when rendering nginx with `--per-ip-rps N` (a positive integer,
default **1**) and `--per-ip-burst N` (default **0**, no extra immediate requests).
For example, `--per-ip-rps 2 --per-ip-burst 1` allows two requests/second and one extra
burst request. The limit is shared across the exposed routes for each peer IP. Re-render,
run `nginx -t`, and reload nginx to apply a change to a running deployment.

The proxy uses nginx's [request limiter](https://nginx.org/en/docs/http/ngx_http_limit_req_module.html)
and [connection limiter](https://nginx.org/en/docs/http/ngx_http_limit_conn_module.html).
Request bodies are buffered at the edge; response buffering is disabled. Credentials belong
in the `Authorization: Bearer ...` or `x-api-key` header. Administration, metrics and generation
routes are private; the initial public product is the decisions API.

## Local preparation

1. Build `surogate-engine` from the validated commit. Copy `surogate-engine` and `libsinfer.so`
   into a release directory, and record their SHA256 sums and the source commit.
2. Keep the artifact at its existing local path. Copy `environment.example` to a local
   `environment` file and fill the release path, secret-file path, actual UID/GID, and immutable
   local Docker image ID. On this box, the tested runtime is the existing `agent-dev:base`
   image; get its ID with `sudo docker image inspect --format '{{.Id}}' agent-dev:base`.
   The runtime image contains the CUDA/FFmpeg libraries used for the build, not the model.
3. Generate a random API key into the configured file with mode 0600, owned by the runtime
   UID. Keep it out of the repository, environment values, command arguments and logs.
4. Run `sudo env RUNE_CONFIG=/absolute/path/environment ./run-engine.sh`. The process holds
   the existing GPU lock, uses the frozen release files and listens on **127.0.0.1:8460**.
   The container mounts the artifact and key read-only. No model is uploaded or copied into an
   image. A busy GPU lock makes startup fail rather than overlap another GPU session.
5. Verify a real authenticated `/v1/decisions` answer. Socket binding or `/v1/models` alone is
   insufficient readiness evidence for benchmarking.

For local TLS checks, render with `--hostname localhost` and a temporary local certificate;
the default listener is **127.0.0.1:8443**. A self-signed staging certificate is only for these
checks. The key is never printed by the provided commands.

## Deployment day

1. Point the chosen hostname at the box and obtain a valid certificate for it. Install nginx
   using the host's package manager. Keep the native engine on loopback.
2. Install `run-engine.sh` and the filled environment at `/etc/rune-v3/`, and the service at
   `/etc/systemd/system/rune-v3.service`. This is already installed and enabled on this box.
   On a fresh installation, stop any manually launched staging container first, then run
   `systemctl daemon-reload` and `systemctl enable --now rune-v3`. The service restarts a
   failed engine; it does not launch alongside a GPU benchmark holding the lock.
3. Render the nginx configuration with explicit public listening and the real certificate:

   ```bash
   python3 render_nginx.py --hostname YOUR_HOSTNAME --listen 0.0.0.0:443 --worker-user www-data \
     --per-ip-rps 1 --per-ip-burst 0 \
     --certificate /path/to/fullchain.pem --private-key /path/to/privkey.pem \
     --prefix /var/lib/rune-v3-proxy --output /etc/nginx/nginx.conf
   nginx -t
   systemctl reload nginx
   ```

   This is a dedicated-box configuration. If nginx is already serving another application,
   integrate its `http`/`server` portions into that configuration instead of replacing it.
   Run the renderer as root for `--worker-user`; it assigns its temporary directories to that user.
4. Check unauthenticated 401, authenticated text/image/thinking decisions, 429 with
   `Retry-After`, and health after a burst through the real hostname. Verify that ports 8460
   and 8443 are not public listeners. Distribute the API key through the normal secret channel.

Manage the engine with `sudo systemctl status rune-v3`, `sudo systemctl restart rune-v3`
and `sudo systemctl stop rune-v3`. Logs are available with `sudo journalctl -u rune-v3 -f`.
Stop the service before GPU benchmarks, and start it again afterwards. Its configuration
is `/etc/rune-v3/environment`; restart the service after changing it. Keep the release and
artifact for rollback; point `RUNE_RELEASE` at a previously validated snapshot and restart the
service. Preserve the local checkpoint, artifacts and benchmark results before retiring the box.

## Validated files on this box

`/home/flavius/work/deployment/rune-v3/` contains the filled environment, private API-key
file and frozen binaries under `releases/`. Each release has `provenance.json` with hashes,
source commit and validation records; `RUNE_RELEASE` in `/etc/rune-v3/environment` selects
the active snapshot. Verification records are under `staging/`. The
`rune-v3.service` systemd unit is installed, enabled at boot, and serves the model on
**127.0.0.1:8460** with API-key authentication. It starts the validated runtime container;
the model remains a read-only mount of the local artifact. The public proxy is still stopped,
so the configurable per-IP proxy policy takes effect when the proxy is activated.
`staging/rate-policy-verification.json` records real nginx checks for the 1/s default and an
override. Do not reuse the short-lived staging certificate for public deployment.
