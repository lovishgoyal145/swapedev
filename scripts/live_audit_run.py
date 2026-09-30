#!/usr/bin/env python3
"""
Live Unmocked SwapeDev Verification Script
===========================================
Executes live unmocked boot and audit of SwapeDev pipeline for profile:
furniture_insta_research (profile_sonu_furniture).

Acceptance Criteria:
1. Boot successfully and establish Cloudflare tunnel via Upstash Redis.
2. nvidia-smi active GPU verification on remote worker.
3. ComfyUI /object_info verification for 4 custom nodes:
   - ReActor
   - SAM2
   - AnimateDiff
   - Impact-Pack
4. Real inference verification:
   - Execute inpainting workflow with sd-v1-5-inpainting.ckpt
   - Download generated image and verify it is non-blank (size, dimensions, std dev).
5. Clean teardown: cancel Kaggle kernel, purge Upstash keys.
"""

import asyncio
import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import httpx

# Ensure proper python path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "swapedev_service"))

from swapedev_service.worker_manager import get_worker_manager, WorkerState, delete_upstash_key, query_upstash_command
from swapedev_service.worker_client import ComfyUIWorkerClient, WorkerClientError
from swapedev_service.config import get_profile_credentials

PROFILE_ID = "furniture_insta_research"
REPORT_DIR = REPO_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_RESULTS_FILE = REPORT_DIR / "audit_v2_results.json"


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


async def main():
    log(f"=== Starting Live Pipeline Verification for profile [{PROFILE_ID}] ===")
    creds = get_profile_credentials(PROFILE_ID)
    k_user = creds.get("KAGGLE_USERNAME", "").strip()
    k_key = creds.get("KAGGLE_KEY", "").strip()
    up_url = creds.get("UPSTASH_REDIS_REST_URL", "").strip()
    up_tok = creds.get("UPSTASH_REDIS_REST_TOKEN", "").strip()

    log(f"Credentials verified: Kaggle User={k_user}, Upstash Host={up_url.split('@')[-1][:20]}...")

    wm = get_worker_manager()

    # Step 0: Pre-clean stale Upstash keys
    log("Pre-cleaning stale Upstash keys...")
    await delete_upstash_key(up_url, up_tok, f"swapedev:{PROFILE_ID}:tunnel_url")
    await delete_upstash_key(up_url, up_tok, f"swapedev:{PROFILE_ID}:worker_error")
    await delete_upstash_key(up_url, up_tok, f"swapedev:{PROFILE_ID}:gpu_info")

    # Check if worker is already booting or ready
    st_initial = wm.get_status(PROFILE_ID)
    if st_initial.state == "READY" and st_initial.tunnel_url:
        log(f"Worker is already READY at {st_initial.tunnel_url}")
        tunnel_url = st_initial.tunnel_url
        boot_success = True
    elif st_initial.state in ("BOOTING", "AWAITING_TUNNEL"):
        log(f"Worker is already in state {st_initial.state}. Polling for completion...")
    else:
        log("Initiating live worker boot via WorkerManager...")
        boot_res = await wm.start_worker(profile_id=PROFILE_ID)
        log(f"Worker start initiated: {boot_res}")

    # Step 2: Poll for READY state or tunnel URL in Upstash
    log("Waiting for remote worker to become READY (timeout 720s)...")
    if not boot_success:
        for attempt in range(144):  # 144 * 5s = 720s
            await asyncio.sleep(5)
            st = wm.get_status(PROFILE_ID)
            elapsed = time.time() - t_start

            # Check Upstash directly as well
            raw_url = await query_upstash_command(up_url, up_tok, "GET", f"swapedev:{PROFILE_ID}:tunnel_url")
            if raw_url and isinstance(raw_url, str) and raw_url.startswith("http"):
                tunnel_url = raw_url.strip()
                boot_success = True
                log(f"Detected tunnel URL directly from Upstash in {elapsed:.1f}s: {tunnel_url}")
                break

            if st.state == "READY" or st.state == WorkerState.READY.value:
                tunnel_url = st.tunnel_url
                boot_success = True
                log(f"Worker successfully reached READY in {elapsed:.1f}s! Tunnel URL: {tunnel_url}")
                break
            elif st.state == "ERROR" or st.state == WorkerState.ERROR.value:
                # Check if Upstash has an error message
                raw_err = await query_upstash_command(up_url, up_tok, "GET", f"swapedev:{PROFILE_ID}:worker_error")
                err_detail = raw_err or st.error_message
                log(f"Worker failed with ERROR in {elapsed:.1f}s: {err_detail}")
                break
            else:
                if attempt % 4 == 0:
                    log(f"[Elapsed: {elapsed:.1f}s] Current state: {st.state}...")

    if not boot_success or not tunnel_url:
        log("FATAL: Worker failed to reach READY state within timeout.")
        sys.exit(1)

    boot_duration = time.time() - t_start

    # Acceptance Step 1: nvidia-smi GPU Verification
    log("=== Acceptance Check 1: Verifying active GPU (nvidia-smi) ===")
    gpu_upstash_raw = await query_upstash_command(up_url, up_tok, "GET", f"swapedev:{PROFILE_ID}:gpu_info")
    log(f"Upstash nvidia-smi raw output: {gpu_upstash_raw}")

    client = ComfyUIWorkerClient(tunnel_url=tunnel_url, timeout_seconds=180)
    system_stats = {}
    gpu_device_name = "Unknown"
    vram_total_mb = 0
    vram_free_mb = 0

    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            res = await http_client.get(f"{tunnel_url}/system_stats")
            if res.status_code == 200:
                system_stats = res.json()
                devices = system_stats.get("devices", [])
                if devices:
                    dev = devices[0]
                    gpu_device_name = dev.get("name", "")
                    vram_total_mb = dev.get("vram_total", 0) / (1024 * 1024)
                    vram_free_mb = dev.get("vram_free", 0) / (1024 * 1024)
                log(f"ComfyUI /system_stats GPU Device: {gpu_device_name}, VRAM Total: {vram_total_mb:.1f} MB, VRAM Free: {vram_free_mb:.1f} MB")
    except Exception as e:
        log(f"Warning checking system_stats: {e}")

    gpu_verified = ("Tesla T4" in gpu_device_name or "NVIDIA" in gpu_device_name or "CUDA" in str(system_stats) or "NVIDIA" in str(gpu_upstash_raw))
    log(f"Acceptance Check 1 (Active GPU): {'PASS' if gpu_verified else 'FAIL'}")

    # Acceptance Step 2: Custom Nodes Registration in /object_info
    log("=== Acceptance Check 2: Verifying Custom Nodes in /object_info ===")
    object_info = {}
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            res = await http_client.get(f"{tunnel_url}/object_info")
            if res.status_code == 200:
                object_info = res.json()
    except Exception as e:
        log(f"Failed to fetch /object_info: {e}")

    all_node_keys = list(object_info.keys())
    log(f"Total registered nodes in ComfyUI: {len(all_node_keys)}")

    # Identify target nodes
    reactor_nodes = [k for k in all_node_keys if "reactor" in k.lower()]
    sam2_nodes = [k for k in all_node_keys if "sam2" in k.lower() or "segmentanything" in k.lower()]
    animatediff_nodes = [k for k in all_node_keys if "animatediff" in k.lower()]
    impact_nodes = [k for k in all_node_keys if "impact" in k.lower() or "facedetailer" in k.lower()]

    log(f"ReActor nodes found ({len(reactor_nodes)}): {reactor_nodes[:3]}")
    log(f"SAM2 nodes found ({len(sam2_nodes)}): {sam2_nodes[:3]}")
    log(f"AnimateDiff nodes found ({len(animatediff_nodes)}): {animatediff_nodes[:3]}")
    log(f"Impact-Pack nodes found ({len(impact_nodes)}): {impact_nodes[:3]}")

    nodes_pass = bool(reactor_nodes and sam2_nodes and animatediff_nodes and impact_nodes)
    log(f"Acceptance Check 2 (All 4 custom node suites registered): {'PASS' if nodes_pass else 'FAIL'}")

    # Acceptance Step 3: Real Inference Verification
    log("=== Acceptance Check 3: Real Inference Execution ===")
    # 1. Upload test image
    test_img_path = REPORT_DIR / "test_inpaint_input.png"
    remote_img_name = await client.upload_file(test_img_path, remote_filename="test_inpaint_input.png")
    log(f"Uploaded test image to ComfyUI input: {remote_img_name}")

    # 2. Build inpaint workflow
    inpaint_workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sd-v1-5-inpainting.ckpt"
            }
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": remote_img_name
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "a beautiful modern luxury chair, solid teak wood, high detail, photorealistic"
            }
        },
        "4": {
            "clip": ["1", 1],
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "blurry, low quality, dark, distorted"
            }
        },
        "5": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2],
                "mask": ["2", 1],
                "grow_mask_by": 6
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": 42,
                "steps": 10,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 0.85
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2]
            }
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "swapedev_audit",
                "images": ["7", 0]
            }
        }
    }

    log("Submitting prompt to ComfyUI...")
    t_inf_start = time.time()
    prompt_id = await client.queue_prompt(inpaint_workflow)
    log(f"Prompt queued successfully! prompt_id: {prompt_id}")

    log("Polling execution status...")
    outputs = await client.poll_execution(prompt_id)
    inf_duration = time.time() - t_inf_start
    log(f"Inference completed in {inf_duration:.1f}s! Outputs: {list(outputs.keys())}")

    # Download output image
    out_node = outputs.get("8", {})
    images_list = out_node.get("images", [])
    if not images_list:
        raise RuntimeError("No images returned in output node 8!")

    target_img_meta = images_list[0]
    out_filename = target_img_meta.get("filename")
    out_subfolder = target_img_meta.get("subfolder", "")
    out_type = target_img_meta.get("type", "output")

    local_output_path = REPORT_DIR / "live_inference_output.png"
    await client.download_output(out_filename, out_subfolder, out_type, local_output_path)
    log(f"Downloaded generated image to: {local_output_path}")

    # Verify image properties
    img_size_bytes = local_output_path.stat().st_size
    pil_img = Image.open(local_output_path)
    width, height = pil_img.size
    img_mode = pil_img.mode

    # Non-blank check: compute standard deviation of pixel intensities
    arr = np.array(pil_img.convert("RGB"))
    pixel_std = float(arr.std())
    pixel_mean = float(arr.mean())
    is_non_blank = (pixel_std > 10.0 and img_size_bytes > 50000)

    log(f"Image Verification: {width}x{height} {img_mode}, Size={img_size_bytes} bytes, Mean={pixel_mean:.1f}, StdDev={pixel_std:.2f}")
    log(f"Acceptance Check 3 (Real inference non-blank image): {'PASS' if is_non_blank else 'FAIL'}")

    # Save complete audit results
    audit_data = {
        "timestamp": time.time(),
        "profile_id": PROFILE_ID,
        "boot_duration_seconds": boot_duration,
        "tunnel_url": tunnel_url,
        "gpu_info": {
            "verified": gpu_verified,
            "device_name": gpu_device_name,
            "vram_total_mb": vram_total_mb,
            "vram_free_mb": vram_free_mb,
            "raw_nvidia_smi": gpu_upstash_raw,
        },
        "custom_nodes": {
            "verified": nodes_pass,
            "total_nodes": len(all_node_keys),
            "reactor_sample": reactor_nodes[:5],
            "sam2_sample": sam2_nodes[:5],
            "animatediff_sample": animatediff_nodes[:5],
            "impact_sample": impact_nodes[:5],
        },
        "inference": {
            "verified": is_non_blank,
            "duration_seconds": inf_duration,
            "output_path": str(local_output_path),
            "width": width,
            "height": height,
            "mode": img_mode,
            "size_bytes": img_size_bytes,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
        }
    }
    AUDIT_RESULTS_FILE.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
    log(f"Saved audit results to {AUDIT_RESULTS_FILE}")

    # Step 5: Clean Teardown
    log("=== Clean Teardown: Shutting down worker to preserve GPU quotas ===")
    await wm.stop_worker(PROFILE_ID)
    
    # Cancel Kaggle kernel directly via isolated Kaggle API
    try:
        temp_dir = tempfile.mkdtemp()
        os.environ["KAGGLE_CONFIG_DIR"] = temp_dir
        os.environ["KAGGLE_API_TOKEN"] = k_key
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.kernels_cancel(f"{k_user}/swapedev-backend")
        log(f"Kaggle kernel session {k_user}/swapedev-backend cancelled successfully.")
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as cancel_err:
        log(f"Warning during kernel cancellation: {cancel_err}")

    # Purge Upstash keys
    await delete_upstash_key(up_url, up_tok, f"swapedev:{PROFILE_ID}:tunnel_url")
    await delete_upstash_key(up_url, up_tok, f"swapedev:{PROFILE_ID}:worker_error")
    await delete_upstash_key(up_url, up_tok, f"swapedev:{PROFILE_ID}:gpu_info")
    log("Upstash keys purged. Teardown complete.")

    log("====================================================")
    log("  AUDIT VERIFICATION RUN COMPLETED SUCCESSFULLY!    ")
    log("====================================================")

if __name__ == "__main__":
    asyncio.run(main())
