import subprocess
from shutil import which
from typing import Union, Optional
import urllib.request
import zipfile
import os
import stat

import lakefs_sdk
from lakefs_sdk import ApiClient, ApiException

RCLONE_DOWNLOAD_URL = "https://downloads.rclone.org/v1.72.0/rclone-v1.72.0-linux-amd64.zip"
RCLONE_BINARY_PATH = "/tmp/rclone"


def ensure_repository(repo_id: str, branch: str, external_id: str, client: ApiClient, type: str):
    """
    Ensure (repo, branch) exists and branch is empty.
    If (repo, branch) exists and branch not empty -> raise.
    If repository missing -> create with 'branch' as default (guaranteed empty).
    If repository exists and branch missing -> create branch; if not empty (because source had objects) raise.
    Note: lakeFS cannot create an *empty* non-default branch if the source reference has objects.
    """
    repos_api = lakefs_sdk.RepositoriesApi(client)

    # 1. Check repository existence
    try:
        repo = repos_api.get_repository(repo_id)
    except ApiException as e:
        if e.status != 404:
            raise
        # Create new repository with desired default (empty)
        creation = lakefs_sdk.RepositoryCreation(
            name=repo_id,
            storage_namespace=f"local://{repo_id}",
            default_branch=branch,
            metadata={"type": type, "externalId": external_id},
        )
        return repos_api.create_repository(creation)

    branches_api = lakefs_sdk.BranchesApi(client)

    # 2. If branch exists, verify emptiness
    try:
        branches = branches_api.list_branches(repo_id, amount=1000)  # adjust amount if many branches expected
    except ApiException:
        raise

    existing_branch_names = {b.id for b in branches.results}
    if branch in existing_branch_names:
        objects_api = lakefs_sdk.ObjectsApi(client)
        objects = objects_api.list_objects(repo_id, branch, amount=1)
        if objects.results:
            raise Exception(f"Repository {repo_id}/{branch} already exists and is not empty.")
        return repo

    # 3. Branch missing: create it
    # Need a source reference; use repository default branch
    src_branch = repo.default_branch
    branches_api.create_branch(
        repo_id,
        lakefs_sdk.BranchCreation(name=branch, source=src_branch),
    )

    # 4. Check emptiness; if not empty, cleanup and raise
    objects_api = lakefs_sdk.ObjectsApi(client)
    objects = objects_api.list_objects(repo_id, branch, amount=1)
    if objects.results:
        # Clean up created branch to honor contract
        try:
            branches_api.delete_branch(repo_id, branch)
        except Exception:
            pass
        raise Exception(
            f"Cannot ensure empty branch '{branch}' because source branch '{src_branch}' has objects."
        )

    return repo


def delete_repository(repo_id: str, branch: str, client: ApiClient) -> bool:
    """
    Delete the repository if (and only if) the specified branch exists, is empty,
    and is the sole branch in the repository.

    If the branch does not exist -> no action.
    If the branch is not empty -> no action.
    If there are other branches -> no action.
    Returns True if the repository was deleted, else False.
    """
    repos_api = lakefs_sdk.RepositoriesApi(client)
    branches_api = lakefs_sdk.BranchesApi(client)
    objects_api = lakefs_sdk.ObjectsApi(client)

    try:
        repos_api.get_repository(repo_id)
    except ApiException as e:
        if e.status == 404:
            return False
        raise

    # Get branches
    try:
        branches = branches_api.list_branches(repo_id, amount=1000)
    except ApiException as e:
        if e.status == 404:
            return False
        raise

    branch_names = {b.id for b in branches.results}
    if branch not in branch_names:
        return False

    # Check emptiness (request only 1 object)
    try:
        objects = objects_api.list_objects(repo_id, branch, amount=1)
    except ApiException as e:
        if e.status == 404:
            return False
        raise

    if objects.results:  # branch not empty
        return False

    # Only delete repository if this is the sole branch
    if len(branch_names) == 1:
        try:
            repos_api.delete_repository(repo_id)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            raise

    # Do nothing if there are other branches (per clarified contract)
    return False


def sync_folder_to_lakefs(
        local_path: str,
        *,
        repo_id: str,
        branch: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        s3_endpoint: Optional[str] = None,
):
    """
    Sync a local folder to lakeFS repository branch using rclone.

    Args:
        local_path: Path to local folder to sync
        repo_id: lakeFS repository ID
        branch: Target branch name
        access_key: lakeFS access key (defaults to env var)
        secret_key: lakeFS secret key (defaults to env var)
        s3_endpoint: lakeFS S3 gateway endpoint
    """
    ensure_rclone_installed()

    env = os.environ.copy()
    env["RCLONE_CONFIG_LAKEFS_TYPE"] = "lakefs"
    if access_key is not None:
        env["RCLONE_CONFIG_LAKEFS_ACCESS_KEY_ID"] = access_key
    if secret_key is not None:
        env["RCLONE_CONFIG_LAKEFS_SECRET_ACCESS_KEY"] = secret_key
    if s3_endpoint is not None:
        env["RCLONE_CONFIG_LAKEFS_ENDPOINT"] = s3_endpoint

    rclone_path = which("rclone") or RCLONE_BINARY_PATH
    command = [rclone_path, "sync", "-L", "--no-check-certificate", local_path, f"lakefs:{repo_id}/{branch}"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result != 0:
        raise Exception(f"rclone sync failed with exit code {result.returncode}: {result.stderr}")


def ensure_rclone_installed():
    if which("rclone") is None:
        # The rclone tgz file contains the rclone binary.
        # Download tha tgz file from RCLONE_DOWNLOAD_URL
        # Decompress and place the rclone binary /tmp
        # Remove the downloaded tgz file
        # Ensure /tmp/rclone is executable
        urllib.request.urlretrieve(RCLONE_DOWNLOAD_URL, "/tmp/rclone.zip")
        with zipfile.ZipFile("/tmp/rclone.zip", 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('/rclone'):
                    with zip_ref.open(file) as source, open(RCLONE_BINARY_PATH, 'wb') as target:
                        target.write(source.read())
                    break

        os.remove("/tmp/rclone.zip")
        os.chmod(RCLONE_BINARY_PATH, os.stat(RCLONE_BINARY_PATH).st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    ensure_rclone_installed()
