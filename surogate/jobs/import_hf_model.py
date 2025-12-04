import os
import sys
import time
from multiprocessing import Queue, Process
from threading import Thread, Event

import lakefs_sdk
import urllib3
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer
from namer import generate as generate_unique_name

from surogate.jobs.progress_reporter import PrinterJobProgressReporter
from surogate.utils.fs import raise_nofile_limit
from surogate.utils.hf import delete_model_from_hf_cache, get_local_snapshot_path, get_downloaded_size_from_cache, \
    check_model_gated, get_repo_file_metadata
from surogate.utils.lakefs import ensure_repository, delete_repository, sync_folder_to_lakefs

ENV_JOB_NAME = "JOB_NAME"
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

job_name = os.getenv(ENV_JOB_NAME, generate_unique_name(category='science'))
hf_model_id = os.getenv(ENV_HF_MODEL_ID, "")
hf_token = os.getenv(ENV_HF_TOKEN, "")
lakefs_repo_id = os.getenv(ENV_LAKEFS_REPO_ID, "")
lakefs_branch = os.getenv(ENV_LAKEFS_BRANCH, "")
lakefs_key = os.getenv(ENV_LAKEFS_KEY, "")
lakefs_secret = os.getenv(ENV_LAKEFS_SECRET, "")
lakefs_endpoint = os.getenv(ENV_LAKEFS_ENDPOINT, "")
progress_reporter = PrinterJobProgressReporter(job_name)


urllib3.disable_warnings()

if not hf_model_id or not lakefs_repo_id or not lakefs_branch or not lakefs_key or not lakefs_secret or not lakefs_endpoint:
    print("ERROR: Missing required environment variables.")
    print(
        f"Make sure {ENV_HF_MODEL_ID}, {ENV_LAKEFS_REPO_ID}, {ENV_LAKEFS_BRANCH}, {ENV_LAKEFS_KEY}, {ENV_LAKEFS_ENDPOINT} and {ENV_LAKEFS_SECRET} are set.")
    exit(1)

# Try to raise the soft limit of open files to reduce chances of hitting 'too many open files'
raise_nofile_limit()

allow_patterns = ["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken",
                  "*.npz", "*.bin", "*.jinja", "*.yaml", "*.md"]

lakefs_configuration = lakefs_sdk.Configuration(host=lakefs_endpoint, username=lakefs_key, password=lakefs_secret)
lakefs_configuration.verify_ssl = False


def cache_progress_monitor(file_metadata, total_bytes):
    """
    Monitor cache directory for download progress.
    Runs in a separate thread.
    """
    global _cache_stop_monitoring

    while not _cache_stop_monitoring:
        try:
            downloaded_bytes = get_downloaded_size_from_cache(hf_model_id, file_metadata, "model")
            progress_reporter.update((downloaded_bytes / total_bytes) * 100)
            progress_reporter.report()

            # Check if download is complete (tolerate minor discrepancies)
            if total_bytes > 0 and downloaded_bytes >= total_bytes * 0.99:
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
        file_metadata, actual_total_size = get_repo_file_metadata(hf_model_id, 'model', allow_patterns)

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


def main():
    global returncode, error_msg

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

            sync_successful = False
            try:
                sync_folder_to_lakefs(local_path, repo_id=lakefs_repo_id, branch=lakefs_branch)
                sync_successful = True

                commitsApi = lakefs_sdk.CommitsApi(api_client)
                commitsApi.commit(
                    repository=lakefs_repo_id,
                    branch=lakefs_branch,
                    commit_creation=lakefs_sdk.CommitCreation(message=f"Add model {hf_model_id}")
                )
            except lakefs_sdk.ApiException as e:
                if e.body and "message" in e.body and "no changes" in e.body["message"].lower():
                    returncode = 0
                else:
                    print(f"ERROR: {e}")
                    returncode = 1
            except Exception as e:
                print(f"ERROR: Failed to sync to lakeFS: {e}")
                returncode = 1

            # Only delete cache if sync was successful
            if sync_successful:
                delete_model_from_hf_cache(hf_model_id)

        if returncode != 0:
            delete_repository(lakefs_repo_id, lakefs_branch, api_client)

    exit(returncode)


if __name__ == "__main__":
    main()
