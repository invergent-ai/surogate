#!/usr/bin/env python3
# Add CUDA tag to wheel version
#
# PyPI/PEP 440 supports local version identifiers (e.g., 0.1.1+cu129)
# This script modifies pyproject.toml to add the CUDA tag to the version
# before building the wheel.
import subprocess
import sys
import tomlkit

CUDA_DEPS = {
    'cu128': [
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cudnn-cu12==9.19.0.56",
        "nvidia-nccl-cu12==2.29.3",
        "nvidia-cufile-cu12==1.14.1.1",
        "nvidia-cuda-nvrtc-cu12==12.8.93"
    ],
    'cu129': [
        "nvidia-cuda-runtime-cu12==12.9.79",
        "nvidia-cudnn-cu12==9.19.0.56",
        "nvidia-nccl-cu12==2.29.3",
        "nvidia-cufile-cu12==1.14.1.1",
        "nvidia-cuda-nvrtc-cu12==12.9.86"
    ],
    'cu13': [
        "nvidia-cuda-runtime==13.1.80",
        "nvidia-cudnn-cu13==9.19.0.56",
        "nvidia-nccl-cu13==2.29.3",
        "nvidia-cufile==1.16.1.26",
        "nvidia-cuda-nvrtc==13.1.115"
    ],
}

def add_cuda_version_tag(cuda_tag: str):
    with open('pyproject.toml', 'r') as f:
        data = tomlkit.load(f)

    # Get current version from git or fallback
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

    # Add CUDA tag to version
    data['project']['version'] = f"{current_version}+{cuda_tag}"
    print(f"Set version to {data['project']['version']}")
    
    existing_deps = list(data['project'].get('dependencies', []))
    data['project']['dependencies'] = existing_deps + CUDA_DEPS[cuda_tag]
    
    with open('pyproject.toml', 'w') as f:
        tomlkit.dump(data, f)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: set_cuda_version_tag.py <cuda_tag>")
        print("Example: set_cuda_version_tag.py cu128")
        sys.exit(1)

    cuda_tag = sys.argv[1]
    add_cuda_version_tag(cuda_tag)
