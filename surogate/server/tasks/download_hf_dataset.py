"""Download a HuggingFace dataset and upload it to the data hub (LakeFS).

Runs as a standalone subprocess spawned by LocalTaskManager.
Configuration is passed via environment variables.

Stdout protocol:
  PROGRESS: <downloaded>/<total>/<pct>%   — parsed by task manager
  ERROR: <message>                        — captured as error_message

Exit codes: 0 = success, 1 = general failure, 77 = gated repo.
"""

import asyncio
import gc
import json
import os
import sys
import time
from multiprocessing import Process, Queue
from threading import Event, Thread

import lakefs_sdk
import urllib3
from datasets import load_dataset
from huggingface_hub import get_paths_info, list_repo_files
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
_stop_monitoring = False


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

def _get_repo_file_metadata(repo_id):
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token or None)
        files = [f for f in files if not f.startswith(".git")]
        path_infos = get_paths_info(repo_id=repo_id, paths=files, repo_type="dataset", token=hf_token or None)
        metadata = {}
        total = 0
        for f, info in zip(files, path_infos):
            size = getattr(info, "size", 0) or 0
            metadata[f] = size
            total += size
        return metadata, total
    except Exception as e:
        print(f"Error getting repo metadata: {e}")
        return {}, 0


def _get_cache_dir(repo_id):
    from huggingface_hub.constants import HF_HUB_CACHE
    return os.path.join(HF_HUB_CACHE, f"datasets--{repo_id.replace('/', '--')}")


def _get_snapshot_path(repo_id):
    snapshots = os.path.join(_get_cache_dir(repo_id), "snapshots")
    if not os.path.isdir(snapshots):
        return None
    commits = sorted(os.listdir(snapshots))
    return os.path.join(snapshots, commits[-1]) if commits else None


def _downloaded_bytes(repo_id, file_metadata):
    snap = _get_snapshot_path(repo_id)
    if not snap:
        return 0
    total = 0
    for name, expected in file_metadata.items():
        path = os.path.join(snap, name)
        if os.path.exists(path):
            total += min(os.path.getsize(path), expected)
    return total


def _progress_monitor(file_metadata, total_bytes):
    global _stop_monitoring
    while not _stop_monitoring:
        try:
            done = _downloaded_bytes(hf_dataset_id, file_metadata)
            pct = (done / total_bytes * 100) if total_bytes > 0 else 0
            print(f"PROGRESS: {done}/{total_bytes}/{pct:.1f}%", flush=True)
            if total_bytes > 0 and done >= total_bytes * 0.99:
                break
            time.sleep(2)
        except Exception:
            time.sleep(5)


def _do_download(repo_id, subset, queue):
    try:
        ds = load_dataset(repo_id, subset or None, token=hf_token or None, keep_in_memory=False)
        try:
            ds.cleanup_cache_files()
        except Exception:
            pass
        queue.put("done")
    except Exception as e:
        queue.put(f"error: {e}")


def _launch_download(repo_id, subset):
    queue = Queue()
    proc = Process(target=_do_download, args=(repo_id, subset, queue))
    proc.start()
    while proc.is_alive():
        proc.join(timeout=0.2)
    result = queue.get() if not queue.empty() else None
    proc.join(timeout=1)
    return result


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
    global returncode, error_msg, _stop_monitoring
    try:
        file_metadata, total = _get_repo_file_metadata(hf_dataset_id)
        monitor = Thread(target=_progress_monitor, args=(file_metadata, total), daemon=True)
        monitor.start()

        result = _launch_download(hf_dataset_id, hf_dataset_subset)
        _stop_monitoring = True
        monitor.join(timeout=5)

        if isinstance(result, str) and result.startswith("error:"):
            returncode = 1
            error_msg = result[len("error: "):]
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

    # 2. Save to LakeFS
    dataset = load_dataset(
        hf_dataset_id, hf_dataset_subset or None,
        token=hf_token or None, keep_in_memory=False,
    )
    try:
        dataset.save_to_disk(
            f"lakefs://{lakefs_repo_id}/{lakefs_branch}",
            storage_options={
                "username": lakefs_key,
                "password": lakefs_secret,
                "host": lakefs_endpoint,
                "verify_ssl": False,
            },
        )

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
