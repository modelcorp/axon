set -ex

# prepare workspace directory
WORKSPACE=$1
if [ -z "$WORKSPACE" ]; then
    export WORKSPACE=$(pwd)/deepgemm_workspace
fi

if [ ! -d "$WORKSPACE" ]; then
    mkdir -p $WORKSPACE
fi

# assume CUDA_HOME is set correctly
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set, please set it to your CUDA installation directory."
    exit 1
fi

is_git_dirty() {
    local dir=$1
    pushd "$dir" > /dev/null

    if [ -d ".git" ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        popd > /dev/null
        return 0  # dirty (true)
    else
        popd > /dev/null
        return 1  # clean (false)
    fi
}

# Function to check if patch has been applied
is_patch_applied() {
    local dir=$1
    local patch_file=$2
    pushd "$dir" > /dev/null
    
    # Try to apply patch in check mode (--check doesn't modify files)
    if git apply --check "$patch_file" 2>/dev/null; then
        # Patch can be applied, so it hasn't been applied yet
        popd > /dev/null
        return 1  # not applied (false)
    else
        # Patch cannot be applied, likely because it's already applied
        # Additional check: see if patch would reverse cleanly
        if git apply --check --reverse "$patch_file" 2>/dev/null; then
            popd > /dev/null
            return 0  # applied (true)
        else
            # Neither forward nor reverse apply cleanly - unclear state
            echo "Warning: Cannot determine patch status for $patch_file"
            popd > /dev/null
            return 1  # assume not applied
        fi
    fi
}

# Function to handle git repository cloning with dirty/incomplete checks
clone_repo() {
    local repo_url=$1
    local dir_name=$2
    local key_file=$3
    local commit_hash=$4

    if [ -d "$dir_name" ]; then
        # Check if directory has uncommitted changes (dirty)
        if is_git_dirty "$dir_name"; then
            echo "$dir_name directory is dirty, skipping clone"
        # Check if clone failed (directory exists but not a valid git repo or missing key files)
        elif [ ! -d "$dir_name/.git" ] || [ ! -f "$dir_name/$key_file" ]; then
            echo "$dir_name directory exists but clone appears incomplete, cleaning up and re-cloning"
            rm -rf "$dir_name"
            git clone --recursive "$repo_url"
            if [ -n "$commit_hash" ]; then
                cd "$dir_name"
                git checkout "$commit_hash"
                cd ..
            fi
        else
            echo "$dir_name directory exists and appears complete; manually update if needed"
        fi
    else
        git clone --recursive "$repo_url"
        if [ -n "$commit_hash" ]; then
            cd "$dir_name"
            git checkout "$commit_hash"
            cd ..
        fi
    fi
}

# build and install deepgemm, require pytorch installed
pushd $WORKSPACE
clone_repo "https://github.com/deepseek-ai/DeepGEMM.git" "DeepGEMM" "setup.py" "79f48ee"
cd DeepGEMM
# Apply patch if not already applied
cp ../../deepgemm.patch .
if is_patch_applied "." "deepgemm.patch"; then
    echo "Patch already applied, skipping"
else
    echo "Applying patch"
    git apply -vvv deepgemm.patch
fi
chmod +x install.sh
./install.sh
popd
