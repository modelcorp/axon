# vLLM Installation for Axon

Axon requires a patched version of vLLM. This directory manages the installation.

## How It Works

- **User mode**: Installs a pinned vLLM release and applies `axon.patch` on top.
- **Dev mode**: Uses an editable checkout of the vLLM fork. No patch needed.

The pinned version, fork URL, fork branch, and checkout path are defined in
`install_vllm.sh`. You can override `VLLM_FORK_URL`, `VLLM_FORK_BRANCH`, and
`VLLM_DEV_PATH` from the environment.

---

## User Install

```bash
cd /path/to/axon
bash install/install.sh
```

This runs all five install stages (CUDA toolkit, PyTorch, Python deps, compiled
extensions, and agent packages) in a single process, ensuring `CUDA_HOME` is
available when building CUDA extensions.

<details>
<summary>Manual step-by-step (advanced)</summary>

```bash
cd /path/to/axon

# 1. Install system/CUDA dependencies
bash install/preinstall_script.sh

# 2. Activate CUDA env vars
source install/utils/source_cuda_env.sh

# 3. Install PyTorch with CUDA 12.8 (must be before Python deps)
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 triton==3.6.0

# 4. Install axon + Python dependencies (does NOT install vllm)
uv pip install -e .

# 5. Install vllm (pinned release + axon patch), flash-attn, apex, megatron-core
bash install/postinstall_script.sh

# 6. Install optional agent packages
uv pip install -e ".[agents]" --no-deps || uv pip install -r install/dependencies/requirements-agents.txt
```

</details>

vllm is installed via `install/vllm/install_vllm.sh`, which:
- Installs the pinned vLLM release
- Applies `axon.patch` to the installed package

---

## Dev Install

```bash
# Option A: let the installer clone the pinned fork into ./vllm
cd /path/to/axon
AXON_VLLM_DEV=1 bash install/install.sh

# Option B: use an existing checkout
git clone https://github.com/modelcorp/vllm.git /path/to/vllm
cd /path/to/vllm
git checkout modelcorp/v0.19.0

cd /path/to/axon
VLLM_DEV_PATH=/path/to/vllm AXON_VLLM_DEV=1 bash install/install.sh
```

Now you can edit vLLM source directly and changes take effect immediately.

---

## Generating the Patch

After making changes on the `modelcorp/<commit>` branch in the vllm fork:

```bash
bash install/vllm/generate_patch.sh /path/to/vllm
```

This diffs `<base_commit>..HEAD` (Python files only) and writes `axon.patch`.
Commit the updated patch file to the axon repo.

---

## Upgrading to a New vllm Version

When a new vLLM release or nightly has features we need (e.g., new model support):

### 1. Find a nightly with the features you need

```bash
# Check what the latest nightly is:
pip install vllm -i https://wheels.vllm.ai/nightly --dry-run 2>&1 | grep "Downloading"
# Output: .../vllm-0.X.Y.devN+gCOMMIT-cp38-abi3-manylinux...whl
```

Note the **commit hash** and **version string** from the wheel filename.

### 2. Update `install_vllm.sh`

Edit these variables at the top of `install/vllm/install_vllm.sh`:

```bash
VLLM_COMMIT="<full 40-char commit hash>"
VLLM_COMMIT_SHORT="<first 9 chars>"
VLLM_VERSION="<version from wheel filename>"
VLLM_WHEEL_URL="vllm==${VLLM_VERSION}"
# Or, for a nightly/specific wheel:
# VLLM_WHEEL_URL="https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-<url-encoded-version>-cp38-abi3-manylinux_2_31_x86_64.whl"
```

### 3. Update `generate_patch.sh`

Edit the `VLLM_BASE_COMMIT` and `VLLM_BASE_SHORT` variables to match.

### 4. Create a new branch in the vllm fork

```bash
cd /path/to/vllm
git fetch origin
git checkout -b modelcorp/<new_commit_short> <new_commit_hash>

# Cherry-pick axon changes from the old branch:
git cherry-pick <old_branch_commit_1> <old_branch_commit_2> ...
# Or rebase:
git rebase --onto <new_commit_hash> <old_base_commit> modelcorp/<old_commit_short>
```

### 5. Regenerate the patch

```bash
bash install/vllm/generate_patch.sh /path/to/vllm
```

### 6. Test

- Run axon inference to verify basic functionality
- Run a short training loop to verify training curves
- Check that MOE routing, MTP logprobs, and pause/continue still work

### 7. Commit

Commit the updated `install_vllm.sh`, `generate_patch.sh`, and `axon.patch` to the axon repo.

---

## Branch Naming Convention

vllm fork branches follow: `modelcorp/v<version>` for releases, `modelcorp/<commit_short>` for nightlies.

Examples:
- `modelcorp/v0.19.0` — based on upstream `releases/v0.19.0` (current)
- `modelcorp/709eadbb0` — based on nightly commit 709eadbb0 (previous, kept as reference)

This makes it clear which upstream version each branch is based on.
