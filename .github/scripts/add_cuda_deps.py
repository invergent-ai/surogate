#!/usr/bin/env python3
# Sigh! pyproject.toml does not provide nice support for dynamic dependencies;
# But we want cuda-12.8 built packages to require 12.8, and 13.0 to require 13.0.
# Seems reasonable, but apparently too much of a niche case for PyPI ?!
# So the solution is to edit the pyproject.toml as part of the build script in the
# workflow :(
#
# This script:
# 1. Sets the package version with CUDA suffix (e.g., 0.1.0+cu128)
# 2. Rewrites torch/torchvision dependencies to use direct URLs from PyTorch index
import subprocess
import sys
import tomlkit

# Mapping of package names to their versions per CUDA tag
# Format: {package_name: {cuda_tag: version}}
TORCH_VERSIONS = {
    'torch': {
        'cu128': '2.9.1',
        'cu129': '2.9.1',
        'cu130': '2.9.1', 
    },
    'torchvision': {
        'cu128': '0.24.1',
        'cu129': '0.24.1',
        'cu130': '0.24.1',
    },
}


def get_direct_url(package_name: str, version: str, index_url: str, cuda_tag: str) -> str:
    """
    Generate a PEP 440 direct URL reference for a PyTorch package.

    Example: torch @ https://download.pytorch.org/whl/cu129/torch-2.9.1%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl
    """
    # PyTorch wheel naming convention
    # Note: + is URL-encoded as %2B in the wheel filename
    wheel_name = f"{package_name}-{version}%2B{cuda_tag}-cp312-cp312-manylinux_2_28_x86_64.whl"
    return f"{package_name} @ {index_url}/{wheel_name}"


def update_dependencies(cuda_tag: str, torch_index_url: str):
    with open('pyproject.toml', 'r') as f:
        data = tomlkit.load(f)

    # Update torch/torchvision dependencies to use direct URLs
    new_deps = []
    for dep in data['project']['dependencies']:
        # Parse dependency name (handle various formats like "torch==2.9.1", "torch>=2.0")
        dep_name = dep.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0].strip()

        if dep_name in TORCH_VERSIONS:
            version = TORCH_VERSIONS[dep_name].get(cuda_tag)
            if not version:
                print(f"Warning: No version found for {dep_name} with {cuda_tag}, skipping")
                new_deps.append(dep)
                continue
            direct_url = get_direct_url(dep_name, version, torch_index_url, cuda_tag)
            new_deps.append(direct_url)
            print(f"Replaced {dep} with {direct_url}")
        else:
            new_deps.append(dep)

    data['project']['dependencies'] = new_deps

    # Add CUDA tag to version
    if 'version' in data['project']:
        current_version = data['project']['version']
    else:
        # First check if we're exactly on a tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match', '--match', 'v*'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            # We're on an exact tag - use it directly
            current_version = result.stdout.strip().lstrip('v')
            print(f"On exact tag, using version: {current_version}")
        else:
            # Not on exact tag, use git describe
            result = subprocess.run(
                ['git', 'describe', '--tags', '--match', 'v*'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                # Convert git describe output (e.g., v0.1.0-5-gabcdef) to PEP 440
                git_version = result.stdout.strip().lstrip('v')
                # Extract base version (before the commit count)
                if '-' in git_version:
                    current_version = git_version.split('-')[0]
                else:
                    current_version = git_version
                print(f"From git describe, using version: {current_version}")
            else:
                # Fallback version from setuptools_scm config
                fallback = data.get('tool', {}).get('setuptools_scm', {}).get('fallback_version', '0.0.1')
                current_version = fallback
                print(f"Using fallback version: {current_version}")

    # Remove 'version' from dynamic list if present
    if 'dynamic' in data['project'] and 'version' in data['project']['dynamic']:
        data['project']['dynamic'].remove('version')
        if not data['project']['dynamic']:
            del data['project']['dynamic']

    data['project']['version'] = f"{current_version}+{cuda_tag}"
    print(f"Set version to {data['project']['version']}")

    with open('pyproject.toml', 'w') as f:
        tomlkit.dump(data, f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: add_cuda_deps.py <cuda_tag> <torch_index_url>")
        print("Example: add_cuda_deps.py cu128 https://download.pytorch.org/whl/cu128")
        sys.exit(1)

    cuda_tag = sys.argv[1]
    torch_index_url = sys.argv[2].rstrip('/')  # Remove trailing slash if present

    update_dependencies(cuda_tag, torch_index_url)
