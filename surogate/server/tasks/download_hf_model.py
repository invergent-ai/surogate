"""Download a HuggingFace model and upload it to the data hub (LakeFS).

Runs as a standalone subprocess spawned by LocalTaskManager.
Configuration is passed via environment variables.

Stdout protocol:
  ERROR: <message>                        — captured as error_message

Exit codes: 0 = success, 1 = general failure, 77 = gated repo.
"""

import asyncio
import json
import os
import sys

import lakefs_sdk
import urllib3
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoConfig, AutoTokenizer

urllib3.disable_warnings()

# ── Configuration from environment ───────────────────────────────────

hf_model_id = os.environ.get("HF_REPO_ID", "")
hf_token = os.environ.get("HF_TOKEN", "")
lakefs_repo_id = os.environ.get("LAKEFS_REPO_ID", "")
lakefs_branch = os.environ.get("LAKEFS_BRANCH", "main")
lakefs_key = os.environ.get("LAKECTL_CREDENTIALS_ACCESS_KEY_ID", "")
lakefs_secret = os.environ.get("LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY", "")
lakefs_endpoint = os.environ.get("LAKECTL_SERVER_ENDPOINT_URL", "")
gguf_file = os.environ.get("GGUF_FILE", "")

# ── Globals ──────────────────────────────────────────────────────────

returncode = 0
error_msg = ""


# ── LakeFS helpers ───────────────────────────────────────────────────

def _lakefs_client():
    endpoint = lakefs_endpoint
    if "/api/v1" not in endpoint:
        endpoint = endpoint.rstrip("/") + "/api/v1"
    cfg = lakefs_sdk.Configuration(
        host=endpoint, username=lakefs_key, password=lakefs_secret,
    )
    cfg.verify_ssl = False
    return lakefs_sdk.ApiClient(cfg)


def _lakefs_commit(message: str):
    with _lakefs_client() as client:
        api = lakefs_sdk.CommitsApi(client)
        api.commit(
            repository=lakefs_repo_id,
            branch=lakefs_branch,
            commit_creation=lakefs_sdk.CommitCreation(message=message),
        )



# ── HuggingFace download ────────────────────────────────────────────

def _get_cache_dir(repo_id):
    from huggingface_hub.constants import HF_HUB_CACHE
    return os.path.join(HF_HUB_CACHE, f"models--{repo_id.replace('/', '--')}")


def _get_snapshot_path(repo_id):
    snapshots = os.path.join(_get_cache_dir(repo_id), "snapshots")
    if not os.path.isdir(snapshots):
        return None
    commits = sorted(os.listdir(snapshots))
    return os.path.join(snapshots, commits[-1]) if commits else None


def _check_gated(repo_id):
    fs = HfFileSystem(token=hf_token or None)
    for name in ("config.json", "model_index.json"):
        try:
            with fs.open(f"{repo_id}/{name}", "r") as f:
                f.read(1)
            return
        except GatedRepoError:
            raise
        except Exception:
            continue


def _delete_cache(repo_id):
    from huggingface_hub import scan_cache_dir
    import shutil
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_type == "model" and repo.repo_id == repo_id:
                shutil.rmtree(repo.repo_path)
                break
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────

def download():
    global returncode, error_msg

    try:
        _check_gated(hf_model_id)
    except GatedRepoError:
        returncode = 77
        error_msg = f"{hf_model_id} is a gated model. Accept the terms on its HuggingFace page first."
        return

    try:
        hf_hub_download(
            repo_id=hf_model_id,
            filename=gguf_file if gguf_file else None,
            token=hf_token or None,
        )
    except Exception as e:
        returncode = 1
        error_msg = f"{type(e).__name__}: {e}"


def _write_chat_template(repo_id, local_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        template = getattr(tokenizer, "chat_template", None)
        if template:
            with open(os.path.join(local_path, "surogate_chat_template.jinja"), "w") as f:
                f.write(template)
    except Exception as e:
        print(f"Warning: Could not extract chat template: {e}")

def _write_model_info(repo_id, local_path):
    try:
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        if config:
            with open(os.path.join(local_path, "surogate_model_info.json"), "w") as f:
                json.dump(config.to_dict(), f, indent=2)
    except Exception as e:
        print(f"Warning: Could not extract model info: {e}")
    
async def _upload_to_lakefs(local_path):
    from anyio import open_process
    from anyio.streams.text import TextReceiveStream

    cmd = f"rclone sync -L --no-check-certificate {local_path} lakefs:{lakefs_repo_id}/{lakefs_branch}"
    proc = await open_process(cmd, stderr=asyncio.subprocess.STDOUT, stdout=asyncio.subprocess.PIPE)
    if proc.stdout:
        async for text in TextReceiveStream(proc.stdout):
            print(f">> {text}")
    await proc.wait()
    return proc.returncode


async def main():
    global returncode

    if not all([hf_model_id, lakefs_repo_id, lakefs_key, lakefs_secret, lakefs_endpoint]):
        print("ERROR: Missing required environment variables.")
        sys.exit(1)

    print(f"HF_MODEL_ID: {hf_model_id}")
    print(f"LAKEFS_REPO_ID: {lakefs_repo_id}/{lakefs_branch}")
    print(f"GGUF_FILE: {gguf_file or '(none)'}")

    # 1. Download from HuggingFace
    download()

    if error_msg:
        print(f"ERROR: {error_msg}")
        sys.exit(returncode)

    local_path = _get_snapshot_path(hf_model_id)
    if not local_path:
        print("ERROR: Download succeeded but snapshot path not found")
        sys.exit(1)

    # 2. Extract chat template for non-GGUF models
    if len(gguf_file) == 0:
        _write_chat_template(hf_model_id, local_path)

    _write_model_info(hf_model_id, local_path)
    
    # 3. Upload to LakeFS via rclone
    rc = await _upload_to_lakefs(local_path)
    if rc != 0:
        print(f"ERROR: rclone exited with code {rc}")
        returncode = 1
    else:
        # 4. Commit
        try:
            _lakefs_commit(f"Import model {hf_model_id}")
        except lakefs_sdk.ApiException as e:
            body = getattr(e, "body", "") or ""
            if "no changes" not in str(body).lower():
                print(f"ERROR: {e}")
                returncode = 1

    # 5. Clean up HF cache
    _delete_cache(hf_model_id)

    sys.exit(returncode)


if __name__ == "__main__":
    asyncio.run(main())
