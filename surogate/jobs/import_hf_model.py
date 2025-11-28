import asyncio
import json
import os
import sys
import time
from asyncio import subprocess
from multiprocessing import Queue, Process
from threading import Thread, Event

import lakefs_sdk
import urllib3
from anyio import open_process
from anyio.streams.text import TextReceiveStream
from huggingface_hub import HfFileSystem, list_repo_files, snapshot_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

from surogate.utils.hf import get_model_architecture, delete_model_from_hf_cache
from surogate.utils.lakefs import ensure_repository, delete_repository

ENV_HF_MODEL_ID = "HF_MODEL_ID"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_LAKEFS_REPO_ID = "LAKEFS_REPO_ID"
ENV_LAKEFS_BRANCH = "LAKEFS_BRANCH"
ENV_LAKEFS_KEY = "LAKEFS_KEY"
ENV_LAKEFS_SECRET = "LAKEFS_SECRET"
ENV_LAKEFS_ENDPOINT = "LAKEFS_ENDPOINT"

# If there is an error set returncode and error_msg
# We're using the following exit codes by convention
# based on Stack Overflow advice:
# 0 = success
# 1 = general failure
# 77 = permission denied (GatedRepoError)
returncode = 0
error_msg = False

# Global variables for cache-based progress tracking
_cache_stop_monitoring = False

hf_model_id = os.getenv(ENV_HF_MODEL_ID, "")
hf_token = os.getenv(ENV_HF_TOKEN, "")
lakefs_repo_id = os.getenv(ENV_LAKEFS_REPO_ID, "")
lakefs_branch = os.getenv(ENV_LAKEFS_BRANCH, "")
lakefs_key = os.getenv(ENV_LAKEFS_KEY, "")
lakefs_secret = os.getenv(ENV_LAKEFS_SECRET, "")
lakefs_endpoint = os.getenv(ENV_LAKEFS_ENDPOINT, "")

urllib3.disable_warnings()

if not hf_model_id or not lakefs_repo_id or not lakefs_branch or not lakefs_key or not lakefs_secret or not lakefs_endpoint:
    print("ERROR: Missing required environment variables.")
    print(f"Make sure {ENV_HF_MODEL_ID}, {ENV_LAKEFS_REPO_ID}, {ENV_LAKEFS_BRANCH}, {ENV_LAKEFS_KEY}, {ENV_LAKEFS_ENDPOINT} and {ENV_LAKEFS_SECRET} are set.")
    exit(1)

allow_patterns = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken",
                  "*.npz", "*.bin", "*.jinja", "*.yaml", "*.md"]

lakefs_configuration = lakefs_sdk.Configuration(host=lakefs_endpoint, username=lakefs_key, password=lakefs_secret)
lakefs_configuration.verify_ssl = False

def get_repo_file_metadata(repo_id, allow_patterns=None):
    """
    Get metadata for all files in a HuggingFace repo.
    Returns dict with filename -> size mapping.
    """
    try:
        # Get list of files in the repo
        files = list_repo_files(repo_id)

        # Filter out git files
        files = [f for f in files if not f.startswith('.git')]

        # Filter by allow_patterns if provided
        if allow_patterns:
            import fnmatch
            filtered_files = []
            for file in files:
                if any(fnmatch.fnmatch(file, pattern) for pattern in allow_patterns):
                    filtered_files.append(file)
            files = filtered_files

        # Get file sizes using HfFileSystem
        fs = HfFileSystem()
        file_metadata = {}
        total_size = 0

        for file in files:
            try:
                # Get file info including size
                file_info = fs.info(f"{repo_id}/{file}")
                file_size = file_info.get('size', 0)
                file_metadata[file] = file_size
                total_size += file_size
            except Exception as e:
                print(f"  Warning: Could not get size for {file}: {e}")
                file_metadata[file] = 0

        return file_metadata, total_size

    except Exception as e:
        print(f"Error getting repo metadata: {e}")
        return {}, 0


def check_model_gated(repo_id):
    """
    Check if a model is gated by trying to read config.json or model_index.json
    using HuggingFace Hub filesystem.

    Args:
        repo_id (str): The repository ID to check

    Raises:
        GatedRepoError: If the model is gated and requires authentication/license acceptance
    """
    fs = HfFileSystem()

    # List of config files to check
    config_files = ["config.json", "model_index.json"]

    # Try to read each config file
    for config_file in config_files:
        file_path = f"{repo_id}/{config_file}"
        try:
            # Try to open and read the file
            with fs.open(file_path, "r") as f:
                f.read(1)  # Just read a byte to check accessibility
            # If we can read any config file, the model is not gated
            return
        except GatedRepoError:
            # If we get a GatedRepoError, the model is definitely gated
            raise GatedRepoError(f"Model {repo_id} is gated and requires authentication or license acceptance")
        except Exception:
            # If we get other errors (like file not found), continue to next file
            continue

    # If we couldn't read any config file due to non-gated errors,
    # we'll let the main download process handle it
    return


def get_cache_dir_for_repo(repo_id):
    """Get the HuggingFace cache directory for a specific repo"""
    from huggingface_hub.constants import HF_HUB_CACHE

    # Convert repo_id to cache-safe name (same logic as huggingface_hub)
    # repo_name = re.sub(r'[^\w\-_.]', '-', repo_id)
    # Replace / with --
    repo_name = repo_id.replace("/", "--")

    return os.path.join(HF_HUB_CACHE, f"models--{repo_name}")

def get_local_snapshot_path(repo_id):
    cache_dir = get_cache_dir_for_repo(repo_id)
    if not os.path.exists(cache_dir):
        return None

    snapshots_dir = os.path.join(cache_dir, "snapshots")
    if not os.path.exists(snapshots_dir):
        return None

    # Get the most recent snapshot (highest timestamp or lexicographically last)
    try:
        commits = os.listdir(snapshots_dir)
        if not commits:
            return 0

        # Use the lexicographically last commit (usually the latest)
        latest_commit = sorted(commits)[-1]
        return os.path.join(snapshots_dir, latest_commit)
    except Exception:
        return None


def get_downloaded_size_from_cache(repo_id, file_metadata):
    """
    Check HuggingFace cache directory to see which files exist and their sizes.
    Returns total downloaded bytes.
    """
    try:
        snapshot_path = get_local_snapshot_path(repo_id)
        if not snapshot_path:
            return 0

        downloaded_size = 0

        # Check each expected file
        for filename, expected_size in file_metadata.items():
            file_path = os.path.join(snapshot_path, filename)

            if os.path.exists(file_path):
                try:
                    actual_size = os.path.getsize(file_path)
                    # Use the smaller of expected and actual size to be conservative
                    downloaded_size += min(actual_size, expected_size)
                except Exception:
                    pass

        return downloaded_size

    except Exception as e:
        print(f"Error checking cache: {e}")
        return 0


def report_progress(downloaded_bytes, total_bytes):
    print(f"PROGRESS: {downloaded_bytes}/{total_bytes}/{(downloaded_bytes / total_bytes) * 100:.1f}%")


def cache_progress_monitor(file_metadata, total_bytes):
    """
    Monitor cache directory for download progress.
    Runs in a separate thread.
    """
    global _cache_stop_monitoring

    while not _cache_stop_monitoring:
        try:
            downloaded_bytes = get_downloaded_size_from_cache(hf_model_id, file_metadata)
            report_progress(downloaded_bytes, total_bytes)

            # Check if download is complete
            if downloaded_bytes >= total_bytes * 0.99:  # 99% complete
                break

            time.sleep(2)  # Check every 2 seconds

        except Exception as e:
            print(f"Error in progress monitor: {e}")
            time.sleep(5)  # Wait longer on error


def do_download(repo_id, queue, allow_patterns=None):
    try:
        # Download without custom progress bar (we'll monitor cache instead)
        snapshot_download(repo_id, allow_patterns=allow_patterns)
        queue.put("done")
    except Exception as e:
        queue.put(f"error: {str(e)}")


def launch_snapshot(repo_id, allow_patterns=None):
    queue = Queue()

    p = Process(target=do_download, args=(repo_id, queue, allow_patterns))
    p.start()

    while p.is_alive():
        sys.stdout.flush()

    result = queue.get()
    return result


def download_blocking(model_is_downloaded):
    global error_msg, returncode, _cache_stop_monitoring

    try:
        check_model_gated(hf_model_id)
    except GatedRepoError:
        returncode = 77
        error_msg = f"{hf_model_id} is a gated HuggingFace model. To continue downloading, you must agree to the terms on the model's HuggingFace page."
        model_is_downloaded.set()
        return

    try:
        # Get file metadata before starting download
        file_metadata, actual_total_size = get_repo_file_metadata(hf_model_id, allow_patterns)

        # Start progress monitoring thread
        progress_thread = Thread(
            target=cache_progress_monitor,
            args=(file_metadata, actual_total_size),
            daemon=True
        )
        progress_thread.start()

        launch_snapshot(repo_id=hf_model_id, allow_patterns=allow_patterns)

        # Stop progress monitoring
        _cache_stop_monitoring = True
        progress_thread.join(timeout=5)

    except GatedRepoError:
        returncode = 77
        error_msg = f"{hf_model_id} is a gated HuggingFace model. \
            To continue downloading, you must agree to the terms \
            on the model's Huggingface page."

    except Exception as e:
        returncode = 1
        error_msg = f"{type(e).__name__}: {e}"

    model_is_downloaded.set()

def write_chat_template(repo_id, local_path):
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    template = getattr(tokenizer, "chat_template", None)
    if template:
        try:
            with open(os.path.join(local_path, "surogate_chat_template.jinja"), "w") as f:
                f.write(template)
        except Exception as e:
            print(f"Warning: Could not write surogate_chat_template.jinja to {local_path}: {e}")

async def main():
    global returncode

    with lakefs_sdk.ApiClient(lakefs_configuration) as api_client:
        try:
            ensure_repository(lakefs_repo_id, lakefs_branch, hf_model_id, api_client, "model")
        except Exception as e:
            print(f"ERROR: {e}")
            exit(1)

    model_is_downloaded = Event()  # A threadsafe flag to coordinate threads
    p2 = Thread(target=download_blocking, args=(model_is_downloaded,))
    p2.start()
    p2.join()

    if error_msg:
        print(f"ERROR: {error_msg}")
    else:
        local_path = get_local_snapshot_path(hf_model_id)

        try:
            write_chat_template(hf_model_id, local_path)
        except:
            print("Warning: Could not extract chat template.")

        try:
            architecture = get_model_architecture(hf_model_id)
            try:
                with open(os.path.join(local_path, "surogate_arch.json"), "w") as f:
                    f.write(json.dumps(architecture, indent=2))
            except Exception as e:
                print(f"Warning: Could not write surogate_arch.json to {local_path}: {e}")
        except:
            print("Warning: Could not extract model architecture.")

        command = f"rclone sync -L --no-check-certificate {local_path} lakefs:{lakefs_repo_id}/{lakefs_branch}"
        process = await open_process(command=command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        if process.stdout:
            async for text in TextReceiveStream(process.stdout):
                print(">> " + text)

        await process.wait()

        if process.returncode != 0:
            print(f"ERROR: rclone exited with code {process.returncode}")
            returncode = 1
        else:
            try:
                commitsApi = lakefs_sdk.CommitsApi(api_client)
                commitsApi.commit(
                    repository=lakefs_repo_id,
                    branch=lakefs_branch,
                    commit_creation=lakefs_sdk.CommitCreation(
                        message="Add model " + hf_model_id
                    )
                )
            except lakefs_sdk.ApiException as e:
                if e.body and "message" in e.body and "no changes" in e.body["message"].lower():
                    returncode = 0
                else:
                    print(f"ERROR: {e}")
                    returncode = 1

        delete_model_from_hf_cache(hf_model_id)

    if returncode != 0:
        with lakefs_sdk.ApiClient(lakefs_configuration) as api_client:
            delete_repository(lakefs_repo_id, lakefs_branch, api_client)

    exit(returncode)


if __name__ == "__main__":
    asyncio.run(main())
