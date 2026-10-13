#!/usr/bin/env bash
# Run INSIDE a Vera Slurm allocation (interactive srun or a batch job).
# Verified configuration: 1x A40, ThinkingCap Qwen3.8 27B FP8, vLLM 0.29.0, MTP, 252k.
set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${TMPDIR:-}" || ! -d "$TMPDIR" ]]; then
  echo "ERROR: Run this inside an allocated Vera Slurm compute job (SLURM_JOB_ID and TMPDIR required)." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export HF_HOME="${HF_HOME:-$TMPDIR/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$TMPDIR/hf-xet}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.config/vera-thinkingcap/hf-token}"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
export APPTAINER_TMPDIR="$TMPDIR/apptainer-tmp"
export TRITON_CACHE_DIR="$TMPDIR/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor-cache"
export VLLM_CACHE_ROOT="$TMPDIR/vllm-cache"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME" \
  "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$(dirname "$HF_TOKEN_PATH")"

# Apptainer typically binds $HOME; explicitly bind the job scratch path.
export APPTAINERENV_HF_HOME="$HF_HOME"
export APPTAINERENV_HF_HUB_CACHE="$HF_HUB_CACHE"
export APPTAINERENV_HF_XET_CACHE="$HF_XET_CACHE"
export APPTAINERENV_HF_TOKEN_PATH="$HF_TOKEN_PATH"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export APPTAINERENV_TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"
export APPTAINERENV_VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT"
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"

SIF="${SIF:-$TMPDIR/vllm-0.29.0.sif}"
MODEL=bottlecapai/ThinkingCap-Qwen3.8-27B-FP8

if [[ ! -s "$HF_TOKEN_PATH" ]]; then
  cat >&2 <<EOF
No persistent Hugging Face token found at:
  $HF_TOKEN_PATH
The model requires accepting its Hugging Face access conditions.
First complete the one-time 'hf auth login' instructions in the guide.
EOF
  exit 2
fi

if [[ ! -s "$SIF" ]]; then
  echo "Pulling vLLM 0.29.0 to $SIF" >&2
  apptainer pull "$SIF" docker://vllm/vllm-openai:v0.29.0
fi

printf 'Job: %s | Node: %s | Scratch: %s\n' "$SLURM_JOB_ID" "$(hostname)" "$TMPDIR"

# Keep the server foregrounded so Slurm owns the service lifetime.
# The model is fetched into HF_HUB_CACHE on first use of a fresh job.
exec apptainer exec --nv --bind "$TMPDIR:$TMPDIR" "$SIF" \
  vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 252000 \
    --gpu-memory-utilization 0.94 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 1 \
    --language-model-only \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
    --enable-prompt-tokens-details \
    --enable-per-request-metrics \
    --kv-cache-metrics \
    --default-chat-template-kwargs '{"reasoning_effort":"xhigh"}'
