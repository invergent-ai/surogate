import gc
import json
import os
import time
from multiprocessing import Queue, Process
from threading import Thread, Event

import lakefs_sdk
import urllib3
from datasets import load_dataset
from huggingface_hub.errors import GatedRepoError
from namer import generate as generate_unique_name

from surogate.jobs.progress_reporter import PrinterJobProgressReporter
from surogate.utils.fs import raise_nofile_limit
from surogate.utils.hf import delete_dataset_from_hf_cache, get_downloaded_size_from_cache, get_repo_file_metadata
from surogate.utils.lakefs import ensure_repository, delete_repository

ENV_JOB_NAME = "JOB_NAME"
ENV_HF_DATASET_ID = "HF_DATASET_ID"
ENV_HF_DATASET_SUBSET = "HF_DATASET_SUBSET"
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
hf_dataset_id = os.getenv(ENV_HF_DATASET_ID, "")
hf_dataset_subset = os.getenv(ENV_HF_DATASET_SUBSET, "")
hf_token = os.getenv(ENV_HF_TOKEN, "")
lakefs_repo_id = os.getenv(ENV_LAKEFS_REPO_ID, "")
lakefs_branch = os.getenv(ENV_LAKEFS_BRANCH, "")
lakefs_key = os.getenv(ENV_LAKEFS_KEY, "")
lakefs_secret = os.getenv(ENV_LAKEFS_SECRET, "")
lakefs_endpoint = os.getenv(ENV_LAKEFS_ENDPOINT, "")
progress_reporter = PrinterJobProgressReporter(job_name)

urllib3.disable_warnings()

allow_patterns = []

lakefs_configuration = lakefs_sdk.Configuration(host=lakefs_endpoint, username=lakefs_key, password=lakefs_secret)
lakefs_configuration.verify_ssl = False

if not hf_dataset_id or not lakefs_repo_id or not lakefs_branch or not lakefs_key or not lakefs_secret or not lakefs_endpoint:
    print("ERROR: Missing required environment variables.")
    print(
        f"Make sure {ENV_HF_DATASET_ID}, {ENV_LAKEFS_REPO_ID}, {ENV_LAKEFS_BRANCH}, {ENV_LAKEFS_KEY}, {ENV_LAKEFS_ENDPOINT} and {ENV_LAKEFS_SECRET} are set.")
    exit(1)

# Try to raise the soft limit of open files to reduce chances of hitting 'too many open files'
raise_nofile_limit()


def cache_progress_monitor(file_metadata, total_bytes):
    """
    Monitor cache directory for download progress.
    Runs in a separate thread.
    """
    global _cache_stop_monitoring

    while not _cache_stop_monitoring:
        try:
            downloaded_bytes = get_downloaded_size_from_cache(hf_dataset_id, file_metadata, "dataset")
            progress_reporter.update((downloaded_bytes / total_bytes) * 100)
            progress_reporter.report()

            # Check if download is complete (tolerate minor discrepancies)
            if total_bytes > 0 and downloaded_bytes >= total_bytes * 0.99:
                break

            time.sleep(2)  # Check every 2 seconds

        except Exception as e:
            print(f"Error in progress monitor: {e}")
            time.sleep(5)  # Wait longer on error


def do_download(repo_id, queue):
    try:
        # keep_in_memory reduces number of simultaneously open mmap'd arrow files
        ds = load_dataset(repo_id, hf_dataset_subset or None, token=hf_token or None, keep_in_memory=False)
        # Explicitly close mmapped files to release descriptors early
        try:
            ds.cleanup_cache_files()
        except Exception:
            pass
        queue.put("done")
    except Exception as e:
        queue.put(f"error: {str(e)}")


def launch_snapshot(repo_id, allow_patterns=None):
    queue = Queue()
    p = Process(target=do_download, args=(repo_id, queue, allow_patterns))
    p.start()

    # Avoid busy spin; poll with small sleep
    while p.is_alive():
        p.join(timeout=0.2)

    result = queue.get() if not queue.empty() else None
    # Ensure process resources are released
    try:
        p.join(timeout=1)
    finally:
        try:
            p.close()
        except Exception:
            pass
    try:
        queue.close()
    except Exception:
        pass

    return result


def download_blocking(is_downloaded):
    global error_msg, returncode, _cache_stop_monitoring

    try:
        file_metadata, actual_total_size = get_repo_file_metadata(hf_dataset_id, 'dataset', allow_patterns)
        progress_thread = Thread(
            target=cache_progress_monitor,
            args=(file_metadata, actual_total_size),
            daemon=True
        )
        progress_thread.start()

        result = launch_snapshot(repo_id=hf_dataset_id, allow_patterns=allow_patterns)
        if isinstance(result, str) and result.startswith("error:"):
            returncode = 1
            error_msg = result[len("error: "):]

        _cache_stop_monitoring = True
        progress_thread.join(timeout=5)

    except GatedRepoError:
        returncode = 77
        error_msg = f"{hf_dataset_id} is a gated HuggingFace dataset.             To continue downloading, you must agree to the terms             on the model's Huggingface page."
    except Exception as e:
        returncode = 1
        error_msg = f"{type(e).__name__}: {e}"

    is_downloaded.set()


def write_dataset_info(dataset, path="/tmp"):
    splits = dataset.shape  # Shape of each split (number of rows, number of columns)
    column_names = dataset.column_names

    splits_info = {}
    for split_name, shape in splits.items():
        splits_info[split_name] = {
            "num_rows": shape[0],
            "column_names": column_names.get(split_name, [])
        }

    with open(f"{path}/surogate_info.json", "w") as f:
        json.dump(splits_info, f, indent=2)


def main():
    global returncode

    with lakefs_sdk.ApiClient(lakefs_configuration) as api_client:
        try:
            ensure_repository(lakefs_repo_id, lakefs_branch, hf_dataset_id, api_client, "dataset")
        except Exception as e:
            print(f"ERROR: {e}")
            exit(1)

    is_downloaded = Event()
    p2 = Thread(target=download_blocking, args=(is_downloaded,))
    p2.start()
    p2.join()

    if error_msg:
        print(f"ERROR: {error_msg}")
    else:
        dataset = load_dataset(hf_dataset_id, hf_dataset_subset or None, token=hf_token or None, keep_in_memory=False)
        try:
            with lakefs_sdk.ApiClient(lakefs_configuration) as api_client:
                try:
                    dataset.save_to_disk(f'lakefs://{lakefs_repo_id}/{lakefs_branch}',
                                         storage_options={"username": lakefs_key, "password": lakefs_secret,
                                                          "host": lakefs_endpoint, "verify_ssl": False})
                    write_dataset_info(dataset)
                    objectsApi = lakefs_sdk.ObjectsApi(api_client)
                    objectsApi.upload_object(
                        repository=lakefs_repo_id,
                        branch=lakefs_branch,
                        path="surogate_info.json",
                        content="/tmp/surogate_info.json"
                    )

                    commitsApi = lakefs_sdk.CommitsApi(api_client)

                    commitsApi.commit(
                        repository=lakefs_repo_id,
                        branch=lakefs_branch,
                        commit_creation=lakefs_sdk.CommitCreation(
                            message="Add dataset " + hf_dataset_id
                        )
                    )
                except lakefs_sdk.ApiException as e:
                    delete_dataset_from_hf_cache(hf_dataset_id)

                    print(f"ERROR: {e}")
                    exit(1)
        except lakefs_sdk.ApiException as e:
            if e.body and "message" in e.body and "no changes" in e.body["message"].lower():
                returncode = 0
            else:
                print(f"ERROR: {e}")
                returncode = 1
        finally:
            try:
                dataset.cleanup_cache_files()
            except Exception:
                pass
            del dataset
            gc.collect()

    delete_dataset_from_hf_cache(hf_dataset_id)

    if returncode != 0:
        with lakefs_sdk.ApiClient(lakefs_configuration) as api_client:
            delete_repository(lakefs_repo_id, lakefs_branch, api_client)

    exit(returncode)


if __name__ == "__main__":
    main()
