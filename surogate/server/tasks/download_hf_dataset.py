"""Download a HuggingFace dataset and upload it to the data hub (LakeFS).

Runs as a standalone subprocess spawned by LocalTaskManager.
Configuration is passed via environment variables.

Stdout protocol:
  ERROR: <message>                        — captured as error_message

Exit codes: 0 = success, 1 = general failure, 77 = gated repo.
"""

import asyncio
import gc
import json
import os
import shutil
import sys
import tempfile

import lakefs_sdk
import urllib3
from datasets import load_dataset
from huggingface_hub.errors import GatedRepoError

urllib3.disable_warnings()

# ── Configuration from environment ───────────────────────────────────

hf_dataset_id = os.environ.get("HF_REPO_ID", "")
hf_dataset_subset = os.environ.get("HF_DATASET_SUBSET", "")
hf_token = os.environ.get("HF_TOKEN", "")
lakefs_repo_id = os.environ.get("LAKEFS_REPO_ID", "")
lakefs_branch = os.environ.get("LAKEFS_BRANCH", "main")
lakefs_key = os.environ.get("LAKECTL_CREDENTIALS_ACCESS_KEY_ID", "")
lakefs_secret = os.environ.get("LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY", "")
lakefs_endpoint = os.environ.get("LAKECTL_SERVER_ENDPOINT_URL", "")

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


def _lakefs_upload(path: str, content_path: str):
    with _lakefs_client() as client:
        api = lakefs_sdk.ObjectsApi(client)
        api.upload_object(
            repository=lakefs_repo_id,
            branch=lakefs_branch,
            path=path,
            content=content_path,
        )


# ── HuggingFace download ────────────────────────────────────────────

def _delete_cache(dataset_id):
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HF_HOME
    import shutil
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_type == "dataset" and repo.repo_id == dataset_id:
                shutil.rmtree(repo.repo_path)
                break
    except Exception:
        pass
    # Also clean the datasets library cache
    try:
        cache2 = os.path.join(HF_HOME, f"datasets/{dataset_id.replace('/', '___')}")
        shutil.rmtree(cache2)
    except Exception:
        pass


def _write_dataset_info(dataset):
    splits = dataset.shape
    column_names = dataset.column_names
    splits_info = {}
    for split_name, shape in splits.items():
        splits_info[split_name] = {
            "num_rows": shape[0],
            "column_names": column_names.get(split_name, []),
            "first_rows": dataset[split_name].select(range(min(3, shape[0]))).to_list(),
        }
    with open("/tmp/surogate_info.json", "w") as f:
        json.dump(splits_info, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────

def download():
    global returncode, error_msg
    try:
        ds = load_dataset(
            hf_dataset_id, hf_dataset_subset or None,
            token=hf_token or None, keep_in_memory=False,
        )
        try:
            ds.cleanup_cache_files()
        except Exception:
            pass
    except GatedRepoError:
        returncode = 77
        error_msg = f"{hf_dataset_id} is a gated dataset. Accept the terms on its HuggingFace page first."
    except Exception as e:
        returncode = 1
        error_msg = f"{type(e).__name__}: {e}"


async def main():
    global returncode

    if not all([hf_dataset_id, lakefs_repo_id, lakefs_key, lakefs_secret, lakefs_endpoint]):
        print("ERROR: Missing required environment variables.")
        sys.exit(1)

    print(f"HF_DATASET_ID: {hf_dataset_id}")
    print(f"HF_DATASET_SUBSET: {hf_dataset_subset or '(default)'}")
    print(f"LAKEFS_REPO_ID: {lakefs_repo_id}/{lakefs_branch}")

    # Raise open file limit
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(soft, 8192), hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

    # 1. Download from HuggingFace
    download()

    if error_msg:
        print(f"ERROR: {error_msg}")
        sys.exit(returncode)

    # 2. Save to LakeFS (via local temp dir to avoid lakefs_spec incompatibility)
    dataset = load_dataset(
        hf_dataset_id, hf_dataset_subset or None,
        token=hf_token or None, keep_in_memory=False,
    )
    tmp_dir = tempfile.mkdtemp(prefix="surogate_ds_")
    try:
        dataset.save_to_disk(tmp_dir)

        # Upload all saved files to LakeFS
        for root, _dirs, files in os.walk(tmp_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, tmp_dir)
                _lakefs_upload(rel_path, local_path)

        # 3. Upload metadata
        _write_dataset_info(dataset)
        _lakefs_upload("surogate_info.json", "/tmp/surogate_info.json")

        # 4. Commit
        _lakefs_commit(f"Import dataset {hf_dataset_id}")

    except lakefs_sdk.ApiException as e:
        body = getattr(e, "body", "") or ""
        if "no changes" not in str(body).lower():
            print(f"ERROR: {e}")
            returncode = 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            dataset.cleanup_cache_files()
        except Exception:
            pass
        del dataset
        gc.collect()

    # 5. Clean up HF cache
    _delete_cache(hf_dataset_id)

    sys.exit(returncode)


if __name__ == "__main__":
    asyncio.run(main())
