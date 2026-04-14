from typing import Optional, List, Union

from sqlalchemy.ext.asyncio import AsyncSession
import lakefs_sdk
from lakefs_sdk import ApiClient, ApiException, RepositoryCreation, RepositoryList, Repository, RefList, Ref, BranchCreation, TagCreation, Commit, CommitList, ObjectStatsList, ObjectStats, AuthApi, UserCreation, Pagination

from surogate.core.config.server_config import ServerConfig
from surogate.core.db.repository.user import set_lakefs_credentials
from surogate.utils.logger import get_logger
import surogate.core.db.repository.user as user_repo

logger = get_logger()

REPO_TYPE_MODEL = "model"
REPO_TYPE_DATASET = "dataset"
REPO_TYPE_AGENT = "agent"
REPO_TYPE_SKILL = "skill"

VALID_REPO_ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9\-\.]{2,62}$"

USERS_GROUP = "Users"
USERS_POLICY = "users-policy"

async def get_lakefs_admin_client(config: ServerConfig) -> ApiClient:
    endpoint = config.lakefs_endpoint
    if "/api/v1" not in endpoint:
        endpoint = endpoint.rstrip("/") + "/api/v1"
    sdk_config = lakefs_sdk.Configuration(host=endpoint, username=config.lakefs_access_key, password=config.lakefs_secret_key)
    sdk_config.verify_ssl = False
    return ApiClient(sdk_config)
    
async def get_lakefs_client(user: str, session: AsyncSession, config: ServerConfig) -> ApiClient:
    endpoint = config.lakefs_endpoint
    if "/api/v1" not in endpoint:
        endpoint = endpoint.rstrip("/") + "/api/v1"

    key, secret = await user_repo.get_lakefs_credentials(session, user)
    sdk_config = lakefs_sdk.Configuration(host=endpoint, username=key, password=secret)
    sdk_config.verify_ssl = False
    return ApiClient(sdk_config)

async def init_lakefs(config: ServerConfig):
    client = await get_lakefs_admin_client(config)
    auth_api = lakefs_sdk.AuthApi(client)
    
    try:
        # Ensure "Users" group exists
        auth_api.get_group(USERS_GROUP)
    except ApiException as e:
        if e.status == 404:
            request = lakefs_sdk.GroupCreation(id=USERS_GROUP)
            auth_api.create_group(group_creation=request)
        else:
            logger.error(f"Error connecting to LakeFS: {e}")
            
    try:
        # Ensure "users-policy" exists and is up to date
        users_policy = lakefs_sdk.Policy(
            id=USERS_POLICY,
            statement=[
                lakefs_sdk.Statement(
                    effect="allow",
                    resource="*",
                    action=["fs:ListRepositories", "fs:CreateRepository", "fs:DeleteRepository", "fs:AttachStorageNamespace"]),
                lakefs_sdk.Statement(
                    effect="allow",
                    resource="*",
                    action=["auth:ListPolicies", "auth:GetPolicy", "auth:CreatePolicy", "auth:UpdatePolicy",
                            "auth:AttachPolicyToUser", "auth:AttachPolicy"])
            ]
        )
        try:
            auth_api.get_policy(USERS_POLICY)
            auth_api.update_policy(USERS_POLICY, policy=users_policy)
        except ApiException as e:
            if e.status == 404:
                auth_api.create_policy(policy=users_policy)
            else:
                raise e
    except ApiException as e:
        logger.error(f"Error connecting to LakeFS: {e}")
            
    # Attach "users-policy" to "Users" group
    try:
        auth_api.attach_policy_to_group(USERS_GROUP, USERS_POLICY)
    except ApiException as e:
        if e.status != 409:  # 409 Conflict means it's already attached
            raise e
        

# ============ Repositories ============

async def list_repositories(client: ApiClient, prefix: Optional[str] = None) -> RepositoryList:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        return repos_api.list_repositories(prefix=prefix)
    except ApiException as e:
        logger.error(f"Error listing LakeFS repositories: {e}")
        return RepositoryList(pagination=Pagination(has_more=False, next_offset="", results=0, max_per_page=0), results=[])
    
async def create_repository(client: ApiClient, user: str, request: RepositoryCreation, config: ServerConfig) -> Optional[Repository]:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        admin_client = await get_lakefs_admin_client(config)
        auth_api = lakefs_sdk.AuthApi(admin_client)
        request.storage_namespace = f"local://{request.name}"
        request.default_branch = "main"
        repo = repos_api.create_repository(repository_creation=request)

        # Create or ensure the repo group exists
        repo_group = f"repo-{request.name}"
        policy_id = f"{request.name}-full-access"
        
        try:
            auth_api.get_group(repo_group)
        except ApiException as e:
            if e.status == 404:
                auth_api.create_group(group_creation=lakefs_sdk.GroupCreation(id=repo_group))
            else:
                await delete_repository(admin_client, request.name, user, config)
                raise e            
            
        # Create or ensure the policy exists
        try:
            auth_api.get_policy(policy_id)
        except ApiException as e:
            if e.status == 404:
                auth_api.create_policy(policy=lakefs_sdk.Policy(
                    id=policy_id, 
                    statement=[
                        lakefs_sdk.Statement(
                            effect="allow",
                            resource=f"arn:lakefs:fs:::repository/{request.name}",
                            action=["fs:*"]),
                         lakefs_sdk.Statement(
                            effect="allow",
                            resource=f"arn:lakefs:fs:::repository/{request.name}/*",
                            action=["fs:*"])
                    ]
                ))
            else:
                await delete_repository(admin_client, request.name, user,config)
                raise e
            
        # Attach the policy to the repo group
        try:
            auth_api.attach_policy_to_group(repo_group, policy_id)
        except ApiException as e:
            if e.status != 409:  # 409 Conflict means it's already attached
                await delete_repository(admin_client, request.name, user, config)
                raise e
        
        # Add repo creator to the group
        auth_api.add_group_membership(repo_group, user)
        
        return repo
    except ApiException as e:
        if e.status == 409:
            raise
        logger.error(f"Error creating LakeFS repository '{request.name}': {e}")
        await delete_repository(admin_client, request.name, user, config)
        return None
    
async def seed_lakefs_user(user: str, session: AsyncSession, config: ServerConfig):
    try:
        existing_creds = await user_repo.get_lakefs_credentials(session, user)
        if existing_creds is not None and all(existing_creds):
            return
        client = await get_lakefs_admin_client(config)
        auth_api = lakefs_sdk.AuthApi(client)
        try:
            auth_api.create_user(user_creation=UserCreation(id=user))
        except ApiException as e:
            if e.status != 409:
                raise
        creds = auth_api.create_credentials(user_id=user)
        try:
            auth_api.add_group_membership(USERS_GROUP, user)
        except ApiException as e:
            if e.status != 409:
                raise
        await set_lakefs_credentials(session, user, creds.access_key_id, creds.secret_access_key)
    except ApiException as e:
        logger.error(f"Error seeding LakeFS user '{user}': {e}")
        return None

async def get_repository(client: ApiClient, repository: str) -> Optional[Repository]:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        return repos_api.get_repository(repository=repository)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS repository '{repository}': {e}")
        return None
    

async def delete_repository(client: ApiClient, repository: str, user: str, config: ServerConfig) -> bool:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)

        repo_group = f"repo-{repository}"
        policy_id = f"{repository}-full-access"

        admin_client = await get_lakefs_admin_client(config)
        auth_api = lakefs_sdk.AuthApi(admin_client)
        
        try:
            auth_api.detach_policy_from_group(repo_group, policy_id)
        except ApiException as e:
            if e.status != 404:
                raise e
        
        try:
            auth_api.delete_group_membership(repo_group, user)
        except ApiException as e:
            if e.status != 404:
                raise e
            
        try:
            auth_api.delete_group(repo_group)
        except ApiException as e:
            if e.status != 404:
                raise e
            
        try:
            auth_api.delete_policy(policy_id)
        except ApiException as e:
            if e.status != 404:
                raise e
            
        repos_api.delete_repository(repository=repository, force=True)
        
        return True
    except ApiException as e:
        if e.status == 404:
            return True
        logger.error(f"Error deleting LakeFS repository '{repository}': {e}")
        return False
    

# ============ Branches ============
async def get_branches(client: ApiClient, repository: str) -> RefList:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        return branches_api.list_branches(repository=repository)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS branches for repository '{repository}': {e}")
        return RefList(pagination=Pagination(has_more=False, next_offset="", results=0, max_per_page=0), results=[])

async def create_branch(client: ApiClient, repository: str, branch: str, source: Optional[str] = None) -> Optional[str]:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        request = BranchCreation(
            name=branch,
            source=source,
            force=False,
            hidden=False
        )
        return branches_api.create_branch(repository=repository, branch_creation=request)
    except ApiException as e:
        logger.error(f"Error creating LakeFS branch '{branch}' in repository '{repository}': {e}")
        return None
    
async def get_branch(client: ApiClient, repository: str, branch: str) -> Optional[Ref]:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        return branches_api.get_branch(repository=repository, branch=branch)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS branch '{branch}' in repository '{repository}': {e}")
        return None

async def delete_branch(client: ApiClient, repository: str, branch: str) -> bool:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        branches_api.delete_branch(repository=repository, branch=branch, force=True)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS branch '{branch}' in repository '{repository}': {e}")
        return False
    
# ============ Tags ============
async def get_tags(client: ApiClient, repository: str) -> RefList:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        return tags_api.list_tags(repository=repository)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS tags for repository '{repository}': {e}")
        return RefList(pagination=Pagination(has_more=False, next_offset="", results=0, max_per_page=0), results=[])

async def create_tag(client: ApiClient, repository: str, tag: str, commit: Optional[str] = None) -> Optional[str]:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        request = TagCreation(
            id=tag,
            ref=commit,
        )
        return tags_api.create_tag(repository=repository, tag_creation=request)
    except ApiException as e:
        logger.error(f"Error creating LakeFS tag '{tag}' in repository '{repository}': {e}")
        return None
    
async def get_tag(client: ApiClient, repository: str, tag: str) -> Optional[Ref]:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        return tags_api.get_tag(repository=repository, tag=tag)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS tag '{tag}' in repository '{repository}': {e}")
        return None
    
async def delete_tag(client: ApiClient, repository: str, tag: str) -> bool:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        tags_api.delete_tag(repository=repository, tag=tag, force=True)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS tag '{tag}' in repository '{repository}': {e}")
        return False
    
# ============ Commits ============
async def get_commits(client: ApiClient, repository: str, ref: str) -> CommitList:
    try:
        commits_api = lakefs_sdk.RefsApi(client)
        return commits_api.log_commits(repository=repository, ref=ref)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS commits for repository '{repository}' and ref '{ref}': {e}")
        return CommitList(pagination=Pagination(has_more=False, next_offset="", results=0, max_per_page=0), results=[])

async def get_commit(client: ApiClient, repository: str, commit_id: str) -> Optional[Commit]:
    try:
        commits_api = lakefs_sdk.CommitsApi(client)
        return commits_api.get_commit(repository=repository, commit_id=commit_id)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS commit '{commit_id}' in repository '{repository}': {e}")
        return None
    
async def commit(client: ApiClient, repository: str, branch: str, message: str = "") -> Optional[str]:
    try:
        commits_api = lakefs_sdk.CommitsApi(client)
        request = lakefs_sdk.CommitCreation(message=message)
        return commits_api.commit(repository=repository, branch=branch, commit_creation=request)
    except ApiException as e:
        logger.error(f"Error creating LakeFS commit in repository '{repository}' and branch '{branch}': {e}")
        return None
    
# ============ Objects ============
async def get_objects(client: ApiClient, repository: str, ref: str, prefix: Optional[str] = None) -> ObjectStatsList:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        return objects_api.list_objects(repository=repository, ref=ref, prefix=prefix, delimiter="/")
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS objects for repository '{repository}' and ref '{ref}': {e}")
        return ObjectStatsList(pagination=Pagination(has_more=False, next_offset="", results=0, max_per_page=0), results=[])

async def delete_object(client: ApiClient, repository: str, branch: str, path: str) -> bool:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        objects_api.delete_object(repository=repository, ref=branch, path=path)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS object '{path}' in repository '{repository}' and branch '{branch}': {e}")
        return False
    
async def stat_object(client: ApiClient, repository: str, ref: str, path: str) -> Optional[ObjectStats]:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        return objects_api.stat_object(repository=repository, ref=ref, path=path)
    except ApiException as e:
        return None
    
async def get_object_content(client: ApiClient, repository: str, ref: str, path: str) -> Optional[bytes]:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        return objects_api.get_object(repository=repository, ref=ref, path=path)
    except ApiException as e:
        return None

async def upload_object(client: ApiClient, repository: str, branch: str, path: str, content: bytes) -> Optional[ObjectStats]:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        return objects_api.upload_object(repository=repository, branch=branch, path=path, content=content)
    except ApiException as e:
        logger.error(f"Error uploading object to LakeFS repository '{repository}' and ref '{branch}': {e}")
        return None
    