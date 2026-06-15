#!/bin/bash
set -e  # Exit on error
set -x  # Print commands

# Search-R1 Training Script for Axon
# Prerequisites:
# 1. Process data: python -m axon.data.preprocess.nq_search
# 2. Setup retrieval index: see scripts/agent/search/setup_retrieval_index.md
#
# This script will automatically start the retrieval server if needed!
#
# Usage:
#   bash train_fsdp.sh [RETRIEVER_NAME] [INDEX_PATH]
#
# Examples:
#   bash train_fsdp.sh                           # Use default (e5-hnsw)
#   bash train_fsdp.sh bm25                      # Use BM25
#   bash train_fsdp.sh e5-flat                   # Use E5 Flat
#   bash train_fsdp.sh e5-hnsw                   # Use E5 HNSW (CPU-friendly)
#   bash train_fsdp.sh e5-hnsw /custom/path.index  # Custom index path

# ============================================================================
# Retrieval Server Configuration
# ============================================================================

# Parse command-line arguments
RETRIEVER_TYPE="${1:-e5-hnsw}"  # Default to e5-hnsw
CUSTOM_INDEX_PATH="${2:-}"

# Map retriever type to configuration
case "$RETRIEVER_TYPE" in
    bm25)
        RETRIEVER_NAME="bm25"
        RETRIEVAL_INDEX_PATH="${CUSTOM_INDEX_PATH:-$HOME/.cache/bm25_index/wiki-18/bm25}"
        RETRIEVAL_CORPUS_PATH="$HOME/.cache/bm25_index/wiki-18/wiki-18.jsonl"
        E5_USE_GPU="false"
        ;;
    e5-flat)
        RETRIEVER_NAME="e5"
        RETRIEVAL_INDEX_PATH="${CUSTOM_INDEX_PATH:-$HOME/.cache/e5_index/wiki-18/e5_Flat.index}"
        RETRIEVAL_CORPUS_PATH="$HOME/.cache/e5_index/wiki-18/wiki-18.jsonl"
        E5_USE_GPU="true"
        echo "WARNING: E5 Flat requires ~60GB GPU memory. Use 'e5-hnsw' for <80GB GPUs."
        ;;
    e5-hnsw)
        RETRIEVER_NAME="e5"
        RETRIEVAL_INDEX_PATH="${CUSTOM_INDEX_PATH:-$HOME/.cache/e5_index/wiki-18-hnsw/e5_HNSW64.index}"
        RETRIEVAL_CORPUS_PATH="$HOME/.cache/e5_index/wiki-18-hnsw/wiki-18.jsonl"
        E5_USE_GPU="false"  # HNSW works well on CPU
        ;;
    *)
        echo "ERROR: Unknown retriever type '$RETRIEVER_TYPE'"
        echo "Valid options: bm25, e5-flat, e5-hnsw"
        exit 1
        ;;
esac

# E5-specific configuration
E5_MODEL="${E5_MODEL:-intfloat/e5-base-v2}"

RETRIEVAL_PORT="${RETRIEVAL_PORT:-8000}"
RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-3}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Auto-start Retrieval Server
# ============================================================================

# Retrieval server conda environment name
RETRIEVAL_CONDA_ENV="${RETRIEVAL_CONDA_ENV:-retrieval_server}"

# Track if we started the server (for cleanup)
STARTED_SERVER=0
SERVER_PID=""

# Function to check if server is running
check_server() {
    curl -s "http://127.0.0.1:${RETRIEVAL_PORT}/health" > /dev/null 2>&1
    return $?
}

# Function to cleanup on exit
cleanup() {
    if [ $STARTED_SERVER -eq 1 ] && [ -n "$SERVER_PID" ]; then
        echo "Stopping retrieval server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        echo "Retrieval server stopped."
    fi
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Check if server is already running
if check_server; then
    echo "✓ ${RETRIEVER_NAME} retrieval server already running at http://127.0.0.1:${RETRIEVAL_PORT}"
else
    echo "Starting ${RETRIEVER_NAME} retrieval server..."
    
    # Check if index exists (different checks for BM25 directory vs E5 file)
    if [ "$RETRIEVER_NAME" = "bm25" ]; then
        if [ ! -d "$RETRIEVAL_INDEX_PATH" ]; then
            echo "ERROR: BM25 index not found at: $RETRIEVAL_INDEX_PATH"
            echo ""
            echo "Please download the BM25 index and corpus. See README.md for instructions."
            echo ""
            echo "Quick setup for BM25:"
            echo "  hf download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir ~/.cache/bm25_index/wiki-18"
            echo "  hf download RUC-NLPIR/FlashRAG_datasets retrieval-corpus/wiki-18.jsonl.gz --repo-type dataset --revision 5d8dc07e0d2f03784f3fcef665110910375a985b --local-dir ~/.cache/bm25_index/wiki-18"
            echo "  cd ~/.cache/bm25_index/wiki-18 && gunzip -k retrieval-corpus/wiki-18.jsonl.gz && mv retrieval-corpus/wiki-18.jsonl ."
            exit 1
        fi
    else
        # E5 uses a file, not a directory
        if [ ! -f "$RETRIEVAL_INDEX_PATH" ]; then
            echo "ERROR: E5 index not found at: $RETRIEVAL_INDEX_PATH"
            echo ""
            echo "Please download the E5 index and corpus. See README.md for instructions."
            echo ""
            echo "Quick setup for E5 (Flat):"
            echo "  mkdir -p ~/.cache/e5_index/wiki-18"
            echo "  # Download from HuggingFace (see README.md for detailed instructions)"
            echo ""
            echo "Or switch to BM25:"
            echo "  export RETRIEVER_NAME=bm25"
            exit 1
        fi
    fi
    
    # Check if corpus exists
    if [ ! -f "$RETRIEVAL_CORPUS_PATH" ]; then
        echo "ERROR: Retrieval corpus not found at: $RETRIEVAL_CORPUS_PATH"
        echo ""
        echo "Please download the corpus. See README.md for instructions."
        exit 1
    fi
    
    # Check if retrieval conda environment exists
    if ! conda env list | grep -q "^${RETRIEVAL_CONDA_ENV} "; then
        echo "ERROR: Conda environment '${RETRIEVAL_CONDA_ENV}' not found"
        echo ""
        echo "Please create the retrieval server environment:"
        echo "  conda env create -f ${SCRIPT_DIR}/retrieval_env.yml"
        echo ""
        echo "Or use a different environment name:"
        echo "  export RETRIEVAL_CONDA_ENV=your_env_name"
        exit 1
    fi
    
    # Build server command based on retriever type
    SERVER_CMD="python3 ${SCRIPT_DIR}/retrieval_server.py \
        --retriever_name $RETRIEVER_NAME \
        --index_path $RETRIEVAL_INDEX_PATH \
        --corpus_path $RETRIEVAL_CORPUS_PATH \
        --port $RETRIEVAL_PORT \
        --topk $RETRIEVAL_TOPK"
    
    # Add E5-specific arguments
    if [ "$RETRIEVER_NAME" = "e5" ]; then
        SERVER_CMD="$SERVER_CMD --retriever_model $E5_MODEL"
        if [ "$E5_USE_GPU" = "true" ]; then
            SERVER_CMD="$SERVER_CMD --faiss_gpu"
        fi
    fi
    
    # Get conda base path
    CONDA_BASE=$(conda info --base)
    
    # Start server in background with conda environment activated
    # We use a subshell to activate conda and run the server
    (
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${RETRIEVAL_CONDA_ENV}"
        eval $SERVER_CMD
    ) > "${SCRIPT_DIR}/retrieval_server.log" 2>&1 &
    
    SERVER_PID=$!
    STARTED_SERVER=1
    
    echo "Waiting for retrieval server to start (PID: $SERVER_PID)..."
    
    # Temporarily disable command printing for the wait loop
    set +x
    
    # Wait up to 600 seconds (10 minutes) for server to be ready
    for i in {1..600}; do
        if check_server; then
            set -x
            echo "✓ Retrieval server ready at http://127.0.0.1:${RETRIEVAL_PORT}"
            break
        fi
        
        # Check if process died
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            set -x
            echo "ERROR: Retrieval server process died. Check logs:"
            echo "  tail ${SCRIPT_DIR}/retrieval_server.log"
            exit 1
        fi
        
        # Print waiting message every 10 seconds
        if [ $((i % 10)) -eq 0 ]; then
            echo "Still waiting for retrieval server... ($i seconds elapsed)"
        fi
        
        sleep 1
    done
    # Re-enable command printing
    set -x
    
    # Final check
    if ! check_server; then
        echo "ERROR: Retrieval server failed to start after 240 seconds"
        echo "Check logs: tail ${SCRIPT_DIR}/retrieval_server.log"
        exit 1
    fi
fi

# ============================================================================
# Training Configuration
# ============================================================================

# Find the directory where axon package is located
AXON_DIR=$(python3 -c "import axon; import os; print(os.path.dirname(os.path.dirname(axon.__file__)))")

# Configurable Arguments
model_path="Qwen/Qwen3-4B-Instruct-2507"  # Default model
loss_mode="${LOSS_MODE:-gspo}"  # Default loss mode
engine="vllm"  # Default engine
enable_partial_rollout=False  # Default enable partial sampler
hybrid_engine=True
step_mode=False # Enable step-level mode (non-cumulative context)
accumulate_history=True # Enable history accumulation

# Extract model name from model path for experiment naming
model_name=$(basename "$model_path")

gpu_memory_utilization=0.85
max_token_len_per_gpu=32768

# Set algorithm-specific parameters
if [ "$loss_mode" = "gspo" ]; then
    clip_ratio_low=0.003
    clip_ratio_high=0.005
    token_reduce=mean
    batch_reduce=step-mean
    experiment_name="${model_name}-search_r1-gspo"
    use_sampler_logprobs=False
else
    # ppo algorithm
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    token_reduce=mean-norm
    batch_reduce=step-mean
    experiment_name="${model_name}-search_r1-grpo++"
    use_sampler_logprobs=False
fi

# Reward configuration (matching Search-R1/train_ppo.sh and reward scoring)
# These control format bonuses during training:
# - structure_format_score: Bonus for valid tag structure
# - final_format_score: Bonus for proper final format
# - retrieval_score: Bonus for retrieving the answer
structure_format_score=0.0
final_format_score=0.0
retrieval_score=0.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Train with PPO
python3 -m axon.driver.train_agent_ppo \
    --config-name='config.yaml' \
    advantage=loop\
    train_files=${AXON_DIR}/data/axon-search-r1/train.parquet \
    val_files=${AXON_DIR}/data/axon-search-r1/test.parquet \
    train_batch_size=512 \
    max_prompt_length=4096 \
    max_seq_length=8192 \
    model_path=${model_path} \
    hybrid_engine=${hybrid_engine} \
    actor.fsdp.use_remove_padding=True \
    actor.fsdp.enable_activation_offload=False \
    actor.fsdp.use_fused_kernels=True \
    actor.fsdp.enable_gradient_checkpointing=True \
    actor.strategy="fsdp2" \
    loss=${loss_mode} \
    loss_args.token_reduce=${token_reduce} \
    loss_args.batch_reduce=${batch_reduce} \
    loss_args.clip_ratio_low=${clip_ratio_low} \
    loss_args.clip_ratio_high=${clip_ratio_high} \
    loss_args.kl_coef=0.0 \
    loss_args.kl_type=low_var_kl \
    loss_args.entropy_coef=0 \
    actor.optimizer_args.lr=1e-6 \
    actor.optimizer_args.weight_decay=0.01 \
    mini_batch_size=128 \
    actor.use_dynamic_bsz=True \
    actor.max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor.fsdp.ulysses_sequence_parallel_size=1 \
    actor.fsdp.grad_norm_threshold=1000 \
    actor.fsdp.offload_policy=False \
    +actor.fsdp.model_dtype="fp32" \
    actor.fsdp.fsdp_size=-1 \
    actor.fsdp.reshard_after_forward=False \
    actor.param_offload=True \
    actor.optimizer_offload=True \
    sampler.tensor_model_parallel_size=1 \
    sampler.enable_prefix_caching=False \
    sampler.name=${engine} \
    sampler.enforce_eager=False \
    decoding.temperature=0.7 \
    decoding.n=8 \
    sampler.gpu_memory_utilization=${gpu_memory_utilization} \
    actor.forward_use_dynamic_bsz=True \
    actor.forward_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor.forward_micro_batch_size_per_gpu=1 \
    ref.param_offload=True \
    ref.forward_micro_batch_size_per_gpu=1 \
    validation.before_train=False \
    validation.steps=20 \
    validation.decoding.n=4 \
    validation.decoding.temperature=0.7 \
    validation.decoding.top_p=0.8 \
    kl_reward_args.kl_coef=0.001 \
    critic.enable=False \
    critic_warmup=0 \
    logger=['console','wandb'] \
    project_name='search-r1-agent' \
    experiment_name=${experiment_name} \
    num_gpus_per_node=8 \
    sampler_trainer_gpu_ratio=1 \
    enable_ray_collective=False \
    num_nodes=1 \
    save_steps=10000 \
    total_epochs=1000 \
    use_sampler_logprobs=${use_sampler_logprobs} \
    +engine_args.disable_thinking=True \
    partial_rollout.enable=${enable_partial_rollout} \
    partial_rollout.n_iters=2 \
    stepwise_advantage_mode="broadcast" \
    program.name=react \
    +program.env_name=${SCRIPT_DIR}/env.py:SearchR1Env \
    +program.env_args.retrieval_url="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve" \
    +program.env_args.terminate_on_incorrect_action=True \
    +program.env_args.topk=${RETRIEVAL_TOPK} \
    +program.env_args.max_turns=5 \
    +program.env_args.structure_format_score=${structure_format_score} \
    +program.env_args.final_format_score=${final_format_score} \
    +program.env_args.retrieval_score=${retrieval_score} \
    +program.agent_name=${SCRIPT_DIR}/agent.py:SearchR1Agent \
    +program.accumulate_history=${accumulate_history} \
    +program.accumulate_thinking=True
