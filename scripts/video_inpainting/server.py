#!/usr/bin/env python3
"""
FastAPI Video Inpainting Service & Webhook Worker for SwapeDev
==============================================================
Exposes headless video character inpainting pipeline over reverse tunnels (pyngrok / cloudflared).
Endpoints:
- POST /process-shot: Inpaints character from input video URL or base64.
- GET  /outputs/{filename}: Direct binary streaming of generated MP4 videos.
- GET  /health: Health check and pipeline readiness.
- GET  /gpu-status: Live CUDA VRAM memory telemetry.
- POST /vram/clear: Manual trigger for PyTorch VRAM eviction.
Features:
- Rigorous CUDA Out-Of-Memory (OOM) handling with auto-recovery.
- Automatic post-run cache clearing (torch.cuda.empty_cache + gc.collect).
- Dual tunnel support: pyngrok & Cloudflare Quick Tunnel.
- Upstash Redis & SwapeDev orchestrator handshake support.
"""

import os
import sys
import gc
import time
import base64
import urllib.request
import urllib.error
import urllib.parse
import json
import logging
import asyncio
import tempfile
import shutil
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("swapedev.video_server")

# Import pipeline module
try:
    from scripts.video_inpainting.pipeline import (
        VideoCharacterInpaintingPipeline,
        flush_vram,
        log_vram_usage,
        TORCH_AVAILABLE,
    )
except ImportError:
    from pipeline import (
        VideoCharacterInpaintingPipeline,
        flush_vram,
        log_vram_usage,
        TORCH_AVAILABLE,
    )

if TORCH_AVAILABLE:
    import torch

# Global Configuration
PORT = int(os.getenv("PORT", "8189"))
HOST = os.getenv("HOST", "0.0.0.0")

def _get_default_dir(kaggle_subpath: str, local_name: str) -> Path:
    if os.path.exists("/kaggle"):
        return Path(f"/kaggle/working/{kaggle_subpath}")
    return Path(tempfile.gettempdir()) / f"swapedev_{local_name}"

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", str(_get_default_dir("outputs", "outputs"))))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", str(_get_default_dir("models", "models"))))
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Global State
active_pipeline: Optional[VideoCharacterInpaintingPipeline] = None
pipeline_lock = asyncio.Lock()
public_tunnel_url: Optional[str] = None


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class ProcessShotRequest(BaseModel):
    video_url: Optional[str] = Field(None, description="Direct HTTP/HTTPS URL of the input video MP4")
    video_base64: Optional[str] = Field(None, description="Base64 encoded input video buffer")
    character_lora: str = Field(
        default="spiderman",
        description="Character LoRA name (e.g. 'spiderman') or path/URL to .safetensors"
    )
    prompt: str = Field(
        default="spiderman suit, highly detailed, cinematic lighting, marvel superhero, 8k",
        description="Target character positive prompt"
    )
    negative_prompt: str = Field(
        default="deformed, blurry, bad anatomy, human face, glitch, boiling artifacts",
        description="Negative prompt"
    )
    denoise: float = Field(
        default=0.80, ge=0.1, le=1.0,
        description="Diffusion denoising strength (0.75-0.85 recommended)"
    )
    num_inference_steps: int = Field(default=25, ge=10, le=50, description="Sampling steps")
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0, description="CFG guidance scale")
    lora_weight: float = Field(default=0.85, ge=0.0, le=2.0, description="LoRA adapter weight")
    seed: Optional[int] = Field(None, description="Deterministic random seed")
    width: int = Field(default=512, description="Target width (vertical mobile aspect)")
    height: int = Field(default=896, description="Target height (vertical mobile aspect)")
    return_format: str = Field(
        default="url",
        description="Response format: 'url', 'base64', or 'both'"
    )
    webhook_url: Optional[str] = Field(None, description="Optional callback URL when shot completes")


class ProcessShotResponse(BaseModel):
    status: str
    video_url: Optional[str] = None
    video_base64: Optional[str] = None
    output_filename: str
    duration_seconds: float
    width: int
    height: int
    error: Optional[str] = None


# ==============================================================================
# FastAPI Initialization
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the pipeline on startup and cleans up on shutdown."""
    global active_pipeline
    logger.info("Initializing SwapeDev Video Inpainting Server...")
    try:
        active_pipeline = VideoCharacterInpaintingPipeline(
            target_width=512,
            target_height=896,
        )
        logger.info("VideoCharacterInpaintingPipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to pre-warm pipeline on startup: {e}")
    yield
    flush_vram()
    logger.info("SwapeDev Video Inpainting Server shutdown complete.")


app = FastAPI(
    title="SwapeDev Video Character Inpainting API",
    description="Headless video character replacement server powered by SAM 2, AnimateDiff V3 & MultiControlNet",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Helper Functions: Video Ingestion & LoRA Resolution
# ==============================================================================

def download_video_from_url(url: str, dest_dir: Path) -> Path:
    """Downloads input video from URL to temporary file."""
    dest_path = dest_dir / f"input_{int(time.time() * 1000)}.mp4"
    logger.info(f"Downloading input video from {url}...")
    headers = {"User-Agent": "SwapeDev-Kaggle-Worker/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    logger.info(f"Downloaded video to {dest_path} ({os.path.getsize(dest_path)} bytes).")
    return dest_path


def resolve_character_lora(lora_param: str) -> Optional[str]:
    """Resolves character LoRA identifier to local file path."""
    if not lora_param:
        return None

    # 1. Direct existing file path
    if os.path.exists(lora_param):
        return lora_param

    # 2. Known preset mappings
    preset_names = {
        "spiderman": "spiderman_classic_sd15.safetensors",
        "spider-man": "spiderman_classic_sd15.safetensors",
        "comic": "spiderman_classic_sd15.safetensors",
    }
    target_filename = preset_names.get(lora_param.lower(), lora_param)

    # Search in checkpoints loras directory
    candidate = CHECKPOINTS_DIR / "loras" / target_filename
    if candidate.exists():
        return str(candidate)

    # Search recursively in checkpoints
    for match in CHECKPOINTS_DIR.glob(f"**/{target_filename}*"):
        if match.is_file():
            return str(match)

    logger.warning(f"Could not resolve local LoRA file for '{lora_param}'. Proceeding without LoRA.")
    return None


# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "SwapeDev Video Character Inpainting",
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available(),
        "tunnel_url": public_tunnel_url,
    }


@app.get("/gpu-status")
async def gpu_status():
    """Returns detailed GPU memory statistics."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"cuda_available": False, "message": "Running on CPU"}

    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3),
    }


@app.post("/vram/clear")
async def manual_clear_vram():
    """Forces VRAM cache flushing and garbage collection."""
    flush_vram()
    return {"status": "ok", "message": "VRAM cache cleared and garbage collection executed."}


@app.get("/outputs/{filename}")
async def get_rendered_video(filename: str):
    """Streams rendered output MP4 video."""
    safe_name = os.path.basename(filename)
    target = OUTPUTS_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File {safe_name} not found")
    return FileResponse(
        str(target),
        media_type="video/mp4",
        filename=safe_name,
    )


@app.post("/process-shot", response_model=ProcessShotResponse)
async def process_shot(request: ProcessShotRequest):
    """
    Main character inpainting endpoint.
    Guarded with strict single-flight execution and comprehensive OOM recovery.
    """
    global active_pipeline
    start_time = time.time()

    if not request.video_url and not request.video_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'video_url' or 'video_base64' must be provided in request body."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="swapedev_req_"))

    # Acquire lock for strict single-flight GPU execution
    async with pipeline_lock:
        try:
            # 1. Stage input video
            if request.video_url:
                input_video_path = download_video_from_url(request.video_url, temp_dir)
            else:
                input_video_path = temp_dir / "input_b64.mp4"
                decoded_bytes = base64.b64decode(request.video_base64)
                with open(input_video_path, "wb") as f:
                    f.write(decoded_bytes)

            # 2. Resolve Character LoRA
            lora_path = resolve_character_lora(request.character_lora)

            # 3. Initialize or reuse pipeline with requested dimensions
            if (
                active_pipeline is None
                or active_pipeline.target_width != request.width
                or active_pipeline.target_height != request.height
            ):
                active_pipeline = VideoCharacterInpaintingPipeline(
                    target_width=request.width,
                    target_height=request.height,
                )

            # 4. Generate Output Path
            output_filename = f"inpainted_{int(time.time() * 1000)}.mp4"
            dest_output_path = OUTPUTS_DIR / output_filename

            # 5. Run inference inside threadpool to keep FastAPI event loop responsive
            logger.info(f"Processing shot with prompt: '{request.prompt}', denoise={request.denoise}")
            rendered_path = await asyncio.to_thread(
                active_pipeline.process_shot,
                input_video_path=input_video_path,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                character_lora_path=lora_path,
                lora_weight=request.lora_weight,
                denoise_strength=request.denoise,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                output_path=dest_output_path,
            )

            # 6. Format Response
            duration = time.time() - start_time
            base_url = public_tunnel_url or f"http://127.0.0.1:{PORT}"
            video_url = f"{base_url.rstrip('/')}/outputs/{output_filename}"

            b64_result = None
            if request.return_format in ("base64", "both"):
                with open(rendered_path, "rb") as vf:
                    b64_result = base64.b64encode(vf.read()).decode("utf-8")

            # Async webhook callback if requested
            if request.webhook_url:
                asyncio.create_task(
                    dispatch_webhook_callback(
                        request.webhook_url,
                        {
                            "status": "completed",
                            "video_url": video_url,
                            "output_filename": output_filename,
                            "duration_seconds": duration,
                        }
                    )
                )

            return ProcessShotResponse(
                status="success",
                video_url=video_url,
                video_base64=b64_result,
                output_filename=output_filename,
                duration_seconds=round(duration, 2),
                width=request.width,
                height=request.height,
            )

        except RuntimeError as rt_err:
            err_str = str(rt_err)
            if "CUDA Out Of Memory" in err_str or "out of memory" in err_str.lower():
                logger.critical(f"CUDA Out Of Memory intercepted: {err_str}")
                flush_vram()
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Kaggle GPU Out Of Memory. Try reducing frame count or resolution. Details: {err_str}"
                )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_str)

        except Exception as e:
            logger.error(f"Error during video character inpainting: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inpainting pipeline error: {str(e)}"
            )

        finally:
            # Absolute VRAM purge and temporary directory cleanup after EVERY run
            flush_vram()
            shutil.rmtree(temp_dir, ignore_errors=True)
            log_vram_usage("Post-Job Final Flush")


async def dispatch_webhook_callback(webhook_url: str, payload: Dict[str, Any]):
    """Delivers async webhook notification to caller."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        await asyncio.to_thread(urllib.request.urlopen, req, timeout=10)
        logger.info(f"Webhook notification delivered to {webhook_url}")
    except Exception as e:
        logger.warning(f"Failed to deliver webhook notification to {webhook_url}: {e}")


# ==============================================================================
# Reverse Tunneling & Orchestrator Discovery
# ==============================================================================

def start_cloudflared_tunnel(port: int) -> Optional[str]:
    """Starts Cloudflare Quick Tunnel and returns public HTTPS URL."""
    cf_binary = "/tmp/cloudflared"
    if not os.path.exists(cf_binary) or not os.access(cf_binary, os.X_OK):
        logger.warning("cloudflared binary not found at /tmp/cloudflared.")
        return None

    cmd = [cf_binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    logger.info(f"Starting Cloudflare Quick Tunnel on port {port}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    start = time.time()
    for line in iter(proc.stdout.readline, ''):
        clean = line.strip()
        match = url_pattern.search(clean)
        if match:
            url = match.group(0)
            logger.info(f"Cloudflare Tunnel active: {url}")
            return url
        if time.time() - start > 45:
            break
    logger.warning("Failed to obtain Cloudflare tunnel URL within timeout.")
    return None


def start_ngrok_tunnel(port: int) -> Optional[str]:
    """Starts pyngrok tunnel if authtoken is present."""
    token = os.getenv("NGROK_AUTHTOKEN")
    if not token:
        logger.info("NGROK_AUTHTOKEN not configured in environment. Skipping ngrok.")
        return None

    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(token)
        tunnel = ngrok.connect(port, bind_tls=True)
        url = tunnel.public_url
        logger.info(f"pyngrok Tunnel active: {url}")
        return url
    except Exception as e:
        logger.warning(f"pyngrok initialization failed: {e}")
        return None


def register_with_swapedev(tunnel_url: str):
    """
    Registers the video worker with SwapeDev orchestrator / Upstash Redis if configured.
    """
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("UPSTASH_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("UPSTASH_REST_TOKEN")
    profile_id = os.getenv("PROFILE_ID", "default_profile")

    if upstash_url and upstash_token:
        logger.info(f"Publishing video worker tunnel URL to Upstash Redis for profile: {profile_id}")
        redis_key = f"swapedev:{profile_id}:video_tunnel_url"
        payload = json.dumps(["SET", redis_key, tunnel_url, "EX", 3600]).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {upstash_token}",
            "Content-Type": "application/json",
        }
        try:
            req = urllib.request.Request(f"{upstash_url.rstrip('/')}/", data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as res:
                logger.info(f"Published Upstash key {redis_key} -> HTTP {res.status}")
        except Exception as e:
            logger.warning(f"Upstash handshake error: {e}")

    # SwapeDev orchestrator direct webhook
    orchestrator_url = os.getenv("SWAPEDEV_ORCHESTRATOR_URL")
    webhook_token = os.getenv("SWAPEDEV_WEBHOOK_TOKEN")
    if orchestrator_url and webhook_token:
        logger.info(f"Sending webhook handshake to SwapeDev orchestrator: {orchestrator_url}")
        handshake_payload = {
            "worker_id": f"kaggle_video_{int(time.time())}",
            "tunnel_url": tunnel_url,
            "profile_id": profile_id,
            "status": "online",
            "gpu_info": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null") or "NVIDIA T4",
            "service_type": "video_inpainting",
            "timestamp": time.time(),
        }
        try:
            req = urllib.request.Request(
                f"{orchestrator_url.rstrip('/')}/api/worker/webhook",
                data=json.dumps(handshake_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {webhook_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as res:
                logger.info(f"SwapeDev orchestrator acknowledged video worker: HTTP {res.status}")
        except Exception as e:
            logger.warning(f"Orchestrator webhook handshake warning: {e}")


# ==============================================================================
# Main Runner with Automatic Tunnel Discovery
# ==============================================================================

def main():
    global public_tunnel_url
    import re
    import uvicorn

    logger.info("==================================================")
    logger.info("  SwapeDev Video Character Inpainting Server     ")
    logger.info("==================================================")

    # 1. Establish reverse tunnel
    # Try pyngrok first if token is available, otherwise Cloudflare Quick Tunnel
    tunnel_url = start_ngrok_tunnel(PORT)
    if not tunnel_url:
        tunnel_url = start_cloudflared_tunnel(PORT)

    if tunnel_url:
        public_tunnel_url = tunnel_url
        logger.info(f"Public ingress URL: {public_tunnel_url}")
        register_with_swapedev(public_tunnel_url)
    else:
        logger.info(f"Running without external tunnel. Bound to http://{HOST}:{PORT}")

    # 2. Run Uvicorn server
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
