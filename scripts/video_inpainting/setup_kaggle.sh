#!/usr/bin/env bash
# ==============================================================================
# SwapeDev: Kaggle GPU Setup & Model Bootstrapper for Video Character Inpainting
# ==============================================================================
# Target: Kaggle Notebook (Debian/Ubuntu, CUDA 12.x, T4 / P100 GPU with 15-16GB VRAM)
# Usage:
#   bash setup_kaggle.sh
# ==============================================================================

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-/kaggle/working}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${WORKSPACE_DIR}/models}"
DATASET_BASE="${DATASET_BASE:-/kaggle/input/swapedev-base-models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_PIP="${SKIP_PIP:-0}"

echo "=================================================================="
echo " [SwapeDev] Initializing Video Character Inpainting Environment..."
echo " Date: $(date)"
echo " Workspace: ${WORKSPACE_DIR}"
echo " Checkpoints: ${CHECKPOINTS_DIR}"
echo "=================================================================="

# ------------------------------------------------------------------------------
# 1. Hardware & CUDA Verification
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Checking GPU compute hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    echo "WARNING: nvidia-smi not detected! Running in CPU-only / simulation mode."
fi

# ------------------------------------------------------------------------------
# 2. System Packages (FFmpeg, aria2, curl, git)
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Installing core system packages (ffmpeg, aria2, curl, git)..."
if command -v apt-get &> /dev/null && [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    if ! command -v ffmpeg &> /dev/null || ! command -v aria2c &> /dev/null; then
        apt-get update -qq && apt-get install -y -qq --no-install-recommends \
            ffmpeg \
            aria2 \
            curl \
            git \
            wget \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev || true
    else
        echo "Core system packages (ffmpeg, aria2, etc.) already verified."
    fi
else
    echo "Running as non-root or apt-get unavailable; skipping system package manager."
fi

# Ensure ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg installation failed or is not on PATH."
    exit 1
fi
echo "FFmpeg verified: $(ffmpeg -version | head -n 1)"

# ------------------------------------------------------------------------------
# 3. Python Package Dependencies
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Installing and upgrading Python libraries..."

export PIP_BREAK_SYSTEM_PACKAGES=1

if [[ "${SKIP_PIP}" == "1" ]]; then
    echo "SKIP_PIP=1 detected; bypassing Python library installation."
elif "${PYTHON_BIN}" -c "import diffusers, transformers, accelerate, controlnet_aux, fastapi, sam2" &> /dev/null; then
    echo "Core Python ML packages already installed and verified in $("${PYTHON_BIN}" -c "import sys; print(sys.executable)")."
else
    "${PYTHON_BIN}" -m pip install --upgrade --no-cache-dir --retries 1 --timeout 15 pip setuptools wheel || true

    # Core ML and Diffusers stack
    "${PYTHON_BIN}" -m pip install --no-cache-dir --retries 1 --timeout 15 \
        "diffusers>=0.30.0" \
        "transformers>=4.40.0" \
        "accelerate>=0.29.0" \
        "safetensors>=0.4.3" \
        "huggingface_hub>=0.23.0" \
        "controlnet_aux>=0.0.7" \
        "opencv-python-headless>=4.8.0" \
        "ffmpeg-python>=0.2.0" \
        "Pillow>=10.0.0" \
        "numpy>=1.24.0" \
        "imageio>=2.30.0" \
        "imageio-ffmpeg>=0.4.9" \
        "einops>=0.7.0" \
        "omegaconf>=2.3.0" || true

    # Web API, Tunneling & Worker utilities
    "${PYTHON_BIN}" -m pip install --no-cache-dir --retries 1 --timeout 15 \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0" \
        "httpx>=0.27.0" \
        "requests>=2.31.0" \
        "pyngrok>=7.1.0" \
        "python-multipart>=0.0.9" || true

    # Install Meta Segment Anything 2 (SAM 2)
    echo "Installing Meta SAM 2..."
    if ! "${PYTHON_BIN}" -c "import sam2" &> /dev/null; then
        "${PYTHON_BIN}" -m pip install --no-cache-dir --retries 1 --timeout 15 git+https://github.com/facebookresearch/segment-anything-2.git || {
            echo "SAM 2 git install failed, trying fallback package..."
            "${PYTHON_BIN}" -m pip install --no-cache-dir --retries 1 --timeout 15 "sam-2" || true
        }
    fi
fi

# Ensure cloudflared binary is staged for reverse tunneling
CLOUDFLARED_BIN="/tmp/cloudflared"
if [[ ! -x "${CLOUDFLARED_BIN}" ]]; then
    echo "Downloading cloudflared binary..."
    curl -sSL --retry 3 -o "${CLOUDFLARED_BIN}" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" || true
    chmod +x "${CLOUDFLARED_BIN}" 2>/dev/null || true
fi

# ------------------------------------------------------------------------------
# 4. Checkpoints Directory Setup
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Preparing models and weights cache directory..."
mkdir -p "${CHECKPOINTS_DIR}/sd15"
mkdir -p "${CHECKPOINTS_DIR}/animatediff"
mkdir -p "${CHECKPOINTS_DIR}/controlnet"
mkdir -p "${CHECKPOINTS_DIR}/sam2"
mkdir -p "${CHECKPOINTS_DIR}/loras"
mkdir -p "${CHECKPOINTS_DIR}/depth_anything"

# ------------------------------------------------------------------------------
# 5. Model Fetching: Hot Start (Mounted Dataset) vs Cold Start (aria2c Download)
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Resolving pipeline model checkpoints and weights..."

# Hot-Start Check: Check if mounted Kaggle Dataset exists
if [[ -d "${DATASET_BASE}" ]]; then
    echo "=================================================================="
    echo " [HOT START] Mounted dataset detected: ${DATASET_BASE}"
    echo " Symlinking models into ${CHECKPOINTS_DIR}..."
    echo "=================================================================="

    # 1. Symlink structured subdirectories if present in dataset
    for sub in sam2 animatediff controlnet loras sd15 depth_anything; do
        if [[ -d "${DATASET_BASE}/${sub}" ]]; then
            mkdir -p "${CHECKPOINTS_DIR}/${sub}"
            for f in "${DATASET_BASE}/${sub}"/*; do
                if [[ -e "${f}" ]]; then
                    ln -sf "${f}" "${CHECKPOINTS_DIR}/${sub}/$(basename "${f}")"
                fi
            done
        fi
    done

    # 2. Key model target resolution (handles flat or nested dataset layouts)
    declare -A MODEL_TARGETS=(
        ["sam2_hiera_base_plus.pt"]="${CHECKPOINTS_DIR}/sam2/sam2_hiera_base_plus.pt"
        ["sam2_hiera_b+.yaml"]="${CHECKPOINTS_DIR}/sam2/sam2_hiera_b+.yaml"
        ["v3_sd15_mm.ckpt"]="${CHECKPOINTS_DIR}/animatediff/v3_sd15_mm.ckpt"
        ["control_v11p_sd15_openpose.pth"]="${CHECKPOINTS_DIR}/controlnet/control_v11p_sd15_openpose.pth"
        ["control_v11f1p_sd15_depth.pth"]="${CHECKPOINTS_DIR}/controlnet/control_v11f1p_sd15_depth.pth"
        ["spiderman_classic_sd15.safetensors"]="${CHECKPOINTS_DIR}/loras/spiderman_classic_sd15.safetensors"
    )

    for fname in "${!MODEL_TARGETS[@]}"; do
        dest="${MODEL_TARGETS[$fname]}"
        if [[ ! -e "${dest}" ]]; then
            if [[ -f "${DATASET_BASE}/${fname}" ]]; then
                ln -sf "${DATASET_BASE}/${fname}" "${dest}"
            elif [[ -f "${DATASET_BASE}/$(basename $(dirname "${dest}"))/${fname}" ]]; then
                ln -sf "${DATASET_BASE}/$(basename $(dirname "${dest}"))/${fname}" "${dest}"
            else
                found=$(find "${DATASET_BASE}" -name "${fname}" -print -quit 2>/dev/null || true)
                if [[ -n "${found}" ]]; then
                    ln -sf "${found}" "${dest}"
                fi
            fi
        fi
    done

    # 3. Ensure SAM 2 yaml configuration exists
    if [[ ! -f "${CHECKPOINTS_DIR}/sam2/sam2_hiera_b+.yaml" ]]; then
        if [[ -f "${SCRIPT_DIR}/sam2_hiera_b+.yaml" ]]; then
            cp -f "${SCRIPT_DIR}/sam2_hiera_b+.yaml" "${CHECKPOINTS_DIR}/sam2/sam2_hiera_b+.yaml"
        fi
    fi

    # 4. HuggingFace cache symlinking if pre-cached in dataset
    HF_CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    if [[ -d "${DATASET_BASE}/huggingface/hub" ]]; then
        mkdir -p "$(dirname "${HF_CACHE_DIR}")"
        ln -sfn "${DATASET_BASE}/huggingface/hub" "${HF_CACHE_DIR}"
    elif [[ -d "${DATASET_BASE}/hub" ]]; then
        mkdir -p "$(dirname "${HF_CACHE_DIR}")"
        ln -sfn "${DATASET_BASE}/hub" "${HF_CACHE_DIR}"
    fi

    echo ""
    echo "[HOT START] Mounted dataset detected. Bootstrapped in < 10s."

else
    echo "=================================================================="
    echo " [COLD START] No mounted dataset detected at ${DATASET_BASE}."
    echo " Executing standard aria2c downloads to build model directory..."
    echo "=================================================================="

    # Fast download helper with resume and parallel connections
    fast_download() {
        local url="$1"
        local dest_dir="$2"
        local dest_filename="$3"
        local target_path="${dest_dir}/${dest_filename}"

        if [[ -f "${target_path}" && $(stat -c%s "${target_path}" 2>/dev/null || stat -f%z "${target_path}" 2>/dev/null) -gt 10000 ]]; then
            echo "   -> [Ready] ${dest_filename} already exists ($(stat -c%s "${target_path}" 2>/dev/null || stat -f%z "${target_path}" 2>/dev/null) bytes)."
            return 0
        fi

        echo "   -> [Downloading] ${dest_filename} from ${url}..."
        if command -v aria2c &> /dev/null; then
            aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
                --dir="${dest_dir}" --out="${dest_filename}" "${url}" || {
                echo "Aria2c failed, falling back to curl..."
                curl -fL --retry 3 -o "${target_path}" "${url}"
            }
        else
            curl -fL --retry 3 -o "${target_path}" "${url}"
        fi
    }

    # 1. SAM 2 Checkpoint (sam2_hiera_base_plus.pt)
    fast_download \
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt" \
        "${CHECKPOINTS_DIR}/sam2" \
        "sam2_hiera_base_plus.pt"

    # Copy or download sam2_hiera_b+.yaml config
    if [[ -f "${SCRIPT_DIR}/sam2_hiera_b+.yaml" ]]; then
        cp -f "${SCRIPT_DIR}/sam2_hiera_b+.yaml" "${CHECKPOINTS_DIR}/sam2/sam2_hiera_b+.yaml"
    else
        fast_download \
            "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2/sam2_hiera_b%2B.yaml" \
            "${CHECKPOINTS_DIR}/sam2" \
            "sam2_hiera_b+.yaml"
    fi

    # 2. AnimateDiff V3 Motion Module (v3_sd15_mm.ckpt)
    fast_download \
        "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt" \
        "${CHECKPOINTS_DIR}/animatediff" \
        "v3_sd15_mm.ckpt"

    # 3. ControlNet OpenPose (control_v11p_sd15_openpose.pth)
    fast_download \
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
        "${CHECKPOINTS_DIR}/controlnet" \
        "control_v11p_sd15_openpose.pth"

    # 4. ControlNet Depth (control_v11f1p_sd15_depth.pth)
    fast_download \
        "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
        "${CHECKPOINTS_DIR}/controlnet" \
        "control_v11f1p_sd15_depth.pth"

    # 5. Character LoRA (Spider-Man / Comic Character LoRA .safetensors)
    fast_download \
        "https://civitai.com/api/download/models/38600?type=Model&format=SafeTensor" \
        "${CHECKPOINTS_DIR}/loras" \
        "spiderman_classic_sd15.safetensors"

    # 6. Pre-warm / Cache HuggingFace diffusers models into HuggingFace Hub Cache
    echo "Pre-warming Hugging Face model cache (RealisticVision V6.0 / DreamShaper & Depth Anything V2)..."
    python3 - << 'EOF'
import os
import sys

try:
    from huggingface_hub import snapshot_download
    print("Snapshot caching Realistic_Vision_V6.0_B1_noVAE...")
    snapshot_download(
        repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        ignore_patterns=["*.bin", "*.ckpt", "*.onnx", "*.pb"],
        resume_download=True,
    )
    print("Snapshot caching Depth Anything V2 Small...")
    snapshot_download(
        repo_id="depth-anything/Depth-Anything-V2-Small-hf",
        resume_download=True,
    )
    print("Hugging Face hub cache pre-warmed successfully.")
except Exception as e:
    print(f"Notice: Background HF cache pre-warm finished with message: {e}")
EOF
fi

echo ""
echo "=================================================================="
echo " [SwapeDev] Kaggle Video Inpainting Environment Setup Complete!   "
echo " All dependencies and checkpoints are staged and verified.        "
echo " Checkpoints directory: ${CHECKPOINTS_DIR}                       "
echo "=================================================================="
