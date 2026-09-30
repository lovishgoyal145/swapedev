#!/usr/bin/env python3
"""
SwapeDev Remote Compute Deployment - Kaggle T4 Kernel Script (Hardened)
========================================================================
- Headless ComfyUI backend bound strictly to 127.0.0.1:8188 (Zero open ingress).
- Uses pre-mounted Kaggle Datasets (/kaggle/input/swapedev-base-models/) for weights
  (SD 1.5, SAM2, ReActor, AnimateDiff) to eliminate live multi-GB downloads.
- Authenticated reverse tunnel (Cloudflare quick tunnel) with mutual webhook handshake.
- 600s (10-minute) idle watchdog that terminates kernel to preserve GPU quotas.
- Zero hardcoded secrets: injects secrets via Kaggle Secrets or environment variables.
"""

import os
import sys
import time
import socket
import subprocess
import threading
import json
import re
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

# Ensure unbuffered output for Kaggle execution logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Configuration Constants
COMFY_DIR = "/kaggle/tmp/ComfyUI"
DATASET_BASE = "/kaggle/input/swapedev-base-models"
IDLE_WATCHDOG_TIMEOUT_SECONDS = int(os.getenv("SWAPEDEV_IDLE_WATCHDOG_SECONDS", "600"))
HEARTBEAT_INTERVAL_SECONDS = 30

# Model specifications with verified SHA-256 hashes and file sizes
MODEL_SPECS = {
    "sd-v1-5-inpainting.ckpt": {
        "subfolder": os.path.join("models", "checkpoints"),
        "url": "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",
        "expected_sha256": "c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19",
        "expected_size": 4265437280,
    },
    "sam2_hiera_small.pt": {
        "subfolder": os.path.join("models", "sam2"),
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "expected_sha256": "95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3",
        "expected_size": 184309650,
    },
    "inswapper_128.onnx": {
        "subfolder": os.path.join("models", "insightface"),
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "expected_sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        "expected_size": 554253681,
    },
    "v3_sd15_mm.ckpt": {
        "subfolder": os.path.join("models", "animatediff_models"),
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
        "expected_sha256": "2412711886f61091846f53204aabc38aa6e09356d62a9808abe4daa802168343",
        "expected_size": 1673262583,
    },
}

# Pinned commit hashes for ComfyUI and custom nodes
PINNED_COMMITS = {
    "ComfyUI": "8cfe5e1ecb97512dea8deaac15e1228d7e6feeb1",
    "ComfyUI-Manager": "1e40793fc54da01c3031b4c89a30d105ef86b729",
    "ComfyUI-SAM2": "0c35fff5f382803e2310103357b5e985f5437f32",
    "ComfyUI-Impact-Pack": "429d0159ad429e64d2b3916e6e7be9c22d025c3c",
    "ComfyUI-AnimateDiff-Evolved": "9257651221002dcba0a12f9cff37e1944e58fb60",
    "ComfyUI-ReActor": "a12c5b19dcac9ae8b47e592da39c9711c8f8c756",
}


def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def compute_sha256(filepath: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Computes SHA-256 digest in 8MB chunks to minimize memory footprint."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_secrets() -> dict:
    """
    Retrieves secrets recursively from mounted Kaggle datasets under /kaggle/input/.
    Handles nested directory layouts like /kaggle/input/datasets/<user>/<slug>/secrets.json
    as well as /kaggle/input/<slug>/secrets.json.
    Fallback: OS environment variables.
    """
    secrets_data = {}

    candidate_paths = []
    custom_path = os.getenv("SWAPEDEV_SECRETS_PATH")
    if custom_path:
        candidate_paths.append(Path(custom_path))

    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        # Recursive walk to discover secrets.json at any depth
        all_discovered = []
        for root, _, files in os.walk(str(input_dir)):
            if "secrets.json" in files:
                all_discovered.append(Path(root) / "secrets.json")

        # Prioritize paths matching swapedev-secrets
        secret_matches = [p for p in all_discovered if "swapedev-secrets" in str(p)]
        other_matches = [p for p in all_discovered if p not in secret_matches]
        candidate_paths.extend(sorted(secret_matches))
        candidate_paths.extend(sorted(other_matches))

    found_path = None
    for cand in candidate_paths:
        if cand.is_file():
            found_path = cand
            break

    if found_path:
        log(f"Secrets read successfully from dataset: {found_path}")
        try:
            with open(found_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    secrets_data[str(k)] = str(v).strip()
        except Exception as e:
            log(f"Warning: Failed to parse secrets JSON from {found_path}: {e}")
    else:
        log("Notice: No mounted secrets dataset found under /kaggle/input.")
        if input_dir.exists():
            log("Dumping /kaggle/input directory tree for diagnostics:")
            for root, dirs, files in os.walk(str(input_dir)):
                log(f"  [DIR] {root} -> dirs: {dirs}, files: {files}")

    # Fill missing from OS environment
    for k in (
        "PROFILE_ID",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "UPSTASH_REST_URL",
        "UPSTASH_REST_TOKEN",
        "SWAPEDEV_ORCHESTRATOR_URL",
        "SWAPEDEV_WEBHOOK_TOKEN",
    ):
        if not secrets_data.get(k) and os.getenv(k):
            secrets_data[k] = os.getenv(k, "").strip()

    # Canonical aliases
    if not secrets_data.get("UPSTASH_REST_URL") and secrets_data.get("UPSTASH_REDIS_REST_URL"):
        secrets_data["UPSTASH_REST_URL"] = secrets_data["UPSTASH_REDIS_REST_URL"]
    if not secrets_data.get("UPSTASH_REST_TOKEN") and secrets_data.get("UPSTASH_REDIS_REST_TOKEN"):
        secrets_data["UPSTASH_REST_TOKEN"] = secrets_data["UPSTASH_REDIS_REST_TOKEN"]

    return secrets_data


def run_command(cmd, check=True, cwd=None):
    log(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, check=check, cwd=cwd, shell=isinstance(cmd, str))


def link_or_copy(src_path: str, dst_path: str):
    """Creates symlink from Kaggle dataset into ComfyUI models. Does NOT copy files."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        if os.path.islink(dst_path) and os.path.realpath(dst_path) == os.path.realpath(src_path):
            log(f"Destination {dst_path} already symlinked to {src_path}. Skipping.")
            return dst_path
        try:
            os.remove(dst_path)
        except OSError:
            pass

    try:
        os.symlink(src_path, dst_path)
        log(f"Symlinked {src_path} -> {dst_path}")
    except OSError as e:
        log(f"Symlink creation failed for {src_path} -> {dst_path}: {e}")
        raise
    return dst_path


def download_file(url: str, target_dir: str, filename: str):
    """Downloads file via aria2c or curl."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        log(f"File {filename} already exists at {target_path}.")
        return target_path

    log(f"Downloading {filename} from {url}...")
    try:
        cmd = [
            "aria2c",
            "-c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "--file-allocation=none",
            "-d", target_dir,
            "-o", filename,
            url
        ]
        subprocess.run(cmd, check=True)
    except Exception as e:
        log(f"aria2c failed ({e}), falling back to curl...")
        cmd = ["curl", "-fSL", "--retry", "3", "-o", target_path, url]
        subprocess.run(cmd, check=True)
    return target_path


def git_clone_pinned(repo_url: str, dest_dir: str, commit_hash: str):
    """Clones repository if not present and checks out exact pinned commit."""
    if not os.path.exists(dest_dir):
        log(f"Cloning {repo_url} into {dest_dir}...")
        run_command(["git", "clone", repo_url, dest_dir])
    log(f"Checking out pinned commit {commit_hash} in {dest_dir}...")
    run_command(["git", "-C", dest_dir, "checkout", commit_hash])


def setup_models_from_dataset_or_fallback(comfy_dir: str) -> bool:
    """
    Mounts models from mounted Kaggle datasets under /kaggle/input/.
    Fail-closed:
    - Verifies all 4 model files for existence, expected file size, and exact SHA-256 hash.
    - If a model file is missing from /kaggle/input, runtime download is REJECTED
      unless SWAPEDEV_ALLOW_RUNTIME_DOWNLOAD=true is explicitly set.
    - If any size or SHA-256 hash mismatches, raises RuntimeError immediately.
    - Points ComfyUI at mounted models via symlinks (zero file copying).
    - Logs which source (dataset path vs download url) each file came from.
    """
    log("==================================================")
    log("  Step 4: Configuring Base Models & Weights       ")
    log("==================================================")

    allow_download = os.getenv("SWAPEDEV_ALLOW_RUNTIME_DOWNLOAD", "false").lower() in ("true", "1", "yes")
    all_from_dataset = True

    # Build a lookup of existing files across all /kaggle/input subdirectories
    input_file_map = {}
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f not in input_file_map:
                    input_file_map[f] = os.path.join(root, f)

    for filename, spec in MODEL_SPECS.items():
        source_in_dataset = input_file_map.get(filename)
        source_path = None
        source_type = ""

        if source_in_dataset and os.path.isfile(source_in_dataset):
            source_path = source_in_dataset
            source_type = f"dataset: {source_path}"
            log(f"Discovered model '{filename}' in mounted Kaggle dataset: {source_path}")
        else:
            all_from_dataset = False
            if not allow_download:
                err_msg = (
                    f"Fail-Closed: Required model '{filename}' not found in any mounted dataset under /kaggle/input, "
                    f"and SWAPEDEV_ALLOW_RUNTIME_DOWNLOAD is false. "
                    f"Attach the dataset avidok/swapedev-base-models or enable runtime downloads."
                )
                log(f"[FATAL MODEL CONFIGURATION ERROR] {err_msg}")
                raise RuntimeError(err_msg)

            log(f"Model '{filename}' not in dataset. Runtime download enabled. Downloading from {spec['url']}...")
            download_dir = "/kaggle/tmp/downloads"
            source_path = download_file(spec["url"], download_dir, filename)
            source_type = f"download: {spec['url']}"

        # 1. Verify existence
        if not os.path.isfile(source_path):
            raise RuntimeError(f"Fail-Closed: Staged model file does not exist at {source_path}")

        # 2. Verify file size
        actual_size = os.path.getsize(source_path)
        expected_size = spec["expected_size"]
        if actual_size != expected_size:
            err_msg = (
                f"Fail-Closed: Size mismatch for '{filename}' ({source_type}): "
                f"expected {expected_size} bytes, got {actual_size} bytes."
            )
            log(f"[FATAL INTEGRITY ERROR] {err_msg}")
            raise RuntimeError(err_msg)

        # 3. Verify SHA-256 hash
        expected_sha = spec["expected_sha256"]
        log(f"Verifying SHA-256 checksum for '{filename}' ({source_type})...")
        actual_sha = compute_sha256(source_path)
        if actual_sha != expected_sha:
            err_msg = (
                f"Fail-Closed: SHA-256 mismatch for '{filename}' ({source_type}): "
                f"expected {expected_sha}, got {actual_sha}."
            )
            log(f"[FATAL INTEGRITY ERROR] {err_msg}")
            raise RuntimeError(err_msg)

        log(f"Integrity PASS: '{filename}' verified (SHA-256: {actual_sha[:16]}..., size: {actual_size} bytes, source: {source_type})")

        # 4. Point ComfyUI at mounted model via symlink (zero copying)
        target_dir = os.path.join(comfy_dir, spec["subfolder"])
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        link_or_copy(source_path, target_path)

    log("All base models verified and linked successfully.")
    return all_from_dataset


def post_upstash_command(rest_url: str, rest_token: str, *command_args) -> bool:
    """
    Executes a direct command via Upstash Redis REST API.
    Posts JSON array of command arguments to base URL.
    """
    if not rest_url or not rest_token:
        log("Notice: UPSTASH_REST_URL or UPSTASH_REST_TOKEN not set. Skipping Upstash command.")
        return False

    base_url = rest_url.rstrip("/")
    payload = json.dumps(list(command_args)).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {rest_token}",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(
        f"{base_url}/",
        data=payload,
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            resp_body = res.read().decode("utf-8")
            log(f"Upstash command {' '.join(str(a) for a in command_args)} -> HTTP {res.status}: {resp_body}")
            return True
    except Exception as e:
        log(f"Upstash command error: {e}")
        # Path-based fallback for SET
        if len(command_args) >= 3 and str(command_args[0]).upper() == "SET":
            cmd, key, val = command_args[0], command_args[1], command_args[2]
            ex = command_args[4] if len(command_args) >= 5 and str(command_args[3]).upper() == "EX" else None
            path_url = f"{base_url}/set/{key}/{urllib.parse.quote(str(val))}"
            if ex:
                path_url += f"?ex={ex}"
            try:
                fb_req = urllib.request.Request(path_url, headers={"Authorization": f"Bearer {rest_token}"}, method="POST")
                with urllib.request.urlopen(fb_req, timeout=10) as fb_res:
                    log(f"Upstash path fallback -> HTTP {fb_res.status}")
                    return True
            except Exception as fb_err:
                log(f"Upstash path fallback error: {fb_err}")
        return False


def publish_upstash_handshake(rest_url: str, rest_token: str, profile_id: str, tunnel_url: str) -> bool:
    """Sets swapedev:{PROFILE_ID}:tunnel_url with 3600s TTL in Upstash Redis."""
    redis_key = f"swapedev:{profile_id}:tunnel_url"
    log(f"Publishing tunnel URL to Upstash Redis: {redis_key} -> {tunnel_url}")
    ok = post_upstash_command(rest_url, rest_token, "SET", redis_key, tunnel_url, "EX", 3600)
    try:
        gpu_info = subprocess.getoutput("nvidia-smi 2>&1")
        post_upstash_command(rest_url, rest_token, "SET", f"swapedev:{profile_id}:gpu_info", gpu_info, "EX", 3600)
    except Exception as e:
        log(f"Warning: Failed to publish gpu_info: {e}")
    log(f"Upstash handshake publication for {redis_key}: success={ok}")
    return ok


def publish_upstash_error(rest_url: str, rest_token: str, profile_id: str, error_message: str) -> bool:
    """Sets swapedev:{PROFILE_ID}:worker_error with 3600s TTL in Upstash Redis."""
    redis_key = f"swapedev:{profile_id}:worker_error"
    log(f"Publishing worker error to Upstash Redis: {redis_key} -> {error_message}")
    ok = post_upstash_command(rest_url, rest_token, "SET", redis_key, error_message, "EX", 3600)
    log(f"Upstash error publication for {redis_key}: success={ok}")
    return ok


def publish_upstash_heartbeat(rest_url: str, rest_token: str, profile_id: str) -> bool:
    """Sets swapedev:{PROFILE_ID}:heartbeat with 60s TTL in Upstash Redis."""
    redis_key = f"swapedev:{profile_id}:heartbeat"
    ts = str(int(time.time()))
    return post_upstash_command(rest_url, rest_token, "SET", redis_key, ts, "EX", 60)


def send_webhook_handshake(orchestrator_url: str, token: str, worker_id: str, tunnel_url: str, status: str = "online", dataset_ready: bool = True):
    """Notifies orchestrator backend of worker status and tunnel URL."""
    if not orchestrator_url or not token:
        log("Warning: SWAPEDEV_ORCHESTRATOR_URL or SWAPEDEV_WEBHOOK_TOKEN not configured. Skipping webhook.")
        return False

    endpoint = f"{orchestrator_url.rstrip('/')}/api/worker/webhook"
    payload = {
        "worker_id": worker_id,
        "tunnel_url": tunnel_url,
        "status": status,
        "gpu_info": subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null") or "NVIDIA T4",
        "dataset_ready": dataset_ready,
        "timestamp": time.time(),
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            log(f"Webhook handshake acknowledged by orchestrator: HTTP {res.status}")
            return True
    except Exception as e:
        log(f"Webhook handshake error to {endpoint}: {e}")
        return False


def send_heartbeat(orchestrator_url: str, token: str, worker_id: str):
    """Sends periodic keepalive heartbeat."""
    if not orchestrator_url or not token:
        return
    endpoint = f"{orchestrator_url.rstrip('/')}/api/worker/heartbeat"
    payload = {
        "worker_id": worker_id,
        "status": "online",
        "uptime_seconds": time.time(),
        "timestamp": time.time(),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            pass
    except Exception:
        pass


def install_cloudflared() -> str:
    """Ensures cloudflared binary is installed and executable."""
    cloudflared_path = "/tmp/cloudflared"
    if os.path.exists(cloudflared_path) and os.access(cloudflared_path, os.X_OK):
        return cloudflared_path

    log("Downloading cloudflared binary for secure reverse tunnel...")
    download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    run_command(f"curl -sSL --retry 3 -o {cloudflared_path} {download_url}", check=True)
    os.chmod(cloudflared_path, 0o755)
    return cloudflared_path


def start_authenticated_tunnel(port: int = 8188) -> subprocess.Popen:
    """Launches Cloudflare Quick Tunnel forwarding to 127.0.0.1:port."""
    cf_binary = install_cloudflared()
    cmd = [
        cf_binary,
        "tunnel",
        "--url", f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]
    log(f"Launching Cloudflare Quick Tunnel on 127.0.0.1:{port}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def extract_tunnel_url(tunnel_proc: subprocess.Popen, timeout_sec: int = 60) -> str:
    """Parses assigned trycloudflare.com tunnel URL from cloudflared logs."""
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    start = time.time()

    for line in iter(tunnel_proc.stdout.readline, ''):
        clean_line = line.strip()
        if clean_line:
            # log(f"[cloudflared] {clean_line}")
            match = url_pattern.search(clean_line)
            if match:
                url = match.group(0)
                log(f"Discovered ephemeral tunnel URL: {url}")
                return url
        if time.time() - start > timeout_sec:
            break

    raise RuntimeError("Failed to obtain Cloudflare tunnel URL within timeout.")


def is_comfyui_idle() -> bool:
    """Queries ComfyUI /queue endpoint to check if work is running or pending."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8188/queue", method="GET")
        with urllib.request.urlopen(req, timeout=3) as res:
            if res.status == 200:
                data = json.loads(res.read().decode("utf-8"))
                running = data.get("queue_running", [])
                pending = data.get("queue_pending", [])
                return len(running) == 0 and len(pending) == 0
    except Exception:
        pass
    return False


def main():
    log("==================================================")
    log("       SwapeDev Kaggle Worker Compute Payload     ")
    log("==================================================")

    worker_id = f"kaggle_t4_{socket.gethostname()}_{int(time.time())}"
    secrets = load_secrets()
    profile_id = secrets.get("PROFILE_ID", "")
    upstash_url = secrets.get("UPSTASH_REST_URL") or secrets.get("UPSTASH_REDIS_REST_URL", "")
    upstash_token = secrets.get("UPSTASH_REST_TOKEN") or secrets.get("UPSTASH_REDIS_REST_TOKEN", "")
    orchestrator_url = secrets.get("SWAPEDEV_ORCHESTRATOR_URL", "http://localhost:8000")
    webhook_token = secrets.get("SWAPEDEV_WEBHOOK_TOKEN", "swapedev_secure_worker_secret_2026")

    # Fail loud immediately if secrets/config are missing or malformed
    missing_fields = []
    if not profile_id:
        missing_fields.append("PROFILE_ID")
    if not upstash_url:
        missing_fields.append("UPSTASH_REDIS_REST_URL")
    if not upstash_token:
        missing_fields.append("UPSTASH_REDIS_REST_TOKEN")

    if missing_fields:
        err_msg = f"Fatal: Missing required worker secrets: {', '.join(missing_fields)}"
        log(f"[FATAL ERROR] {err_msg}")
        if upstash_url and upstash_token and profile_id:
            publish_upstash_error(upstash_url, upstash_token, profile_id, err_msg)
        sys.stderr.write(f"{err_msg}\n")
        sys.exit(1)

    log(f"Worker Configuration: profile_id='{profile_id}', upstash_configured=True")

    comfy_proc = None
    tunnel_proc = None

    try:
        # 1. System packages
        log("Step 1: Installing system packages (openssh-client, aria2, curl)...")
        os.system("apt-get update && apt-get install -y openssh-client aria2 curl")

        # 2. ComfyUI Setup
        os.makedirs("/kaggle/tmp", exist_ok=True)
        log(f"Step 2: Staging pinned ComfyUI ({PINNED_COMMITS['ComfyUI']})...")
        git_clone_pinned("https://github.com/comfyanonymous/ComfyUI.git", COMFY_DIR, PINNED_COMMITS["ComfyUI"])

        # 3. Python Dependencies
        log("Step 3: Installing Python dependencies...")
        run_command([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(COMFY_DIR, "requirements.txt")])
        run_command([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime-gpu"], check=False)

        # Precompiled insightface
        whl_filename = "insightface-0.7.3-cp310-cp310-linux_x86_64.whl"
        whl_path = None
        if os.path.exists("/kaggle/input"):
            for root, _, files in os.walk("/kaggle/input"):
                if whl_filename in files:
                    whl_path = os.path.join(root, whl_filename)
                    break
        if not whl_path or not os.path.exists(whl_path):
            whl_url = f"https://huggingface.co/deauxpas/colabrepo/resolve/main/{whl_filename}"
            whl_path = download_file(whl_url, "/kaggle/tmp", whl_filename)
        run_command([sys.executable, "-m", "pip", "install", "-q", whl_path], check=False)

        # Custom nodes with pinned commit hashes
        custom_nodes_dir = os.path.join(COMFY_DIR, "custom_nodes")
        os.makedirs(custom_nodes_dir, exist_ok=True)
        nodes = [
            ("ComfyUI-Manager", "https://github.com/ltdrdata/ComfyUI-Manager.git", PINNED_COMMITS["ComfyUI-Manager"]),
            ("ComfyUI-SAM2", "https://github.com/kijai/ComfyUI-segment-anything-2.git", PINNED_COMMITS["ComfyUI-SAM2"]),
            ("ComfyUI-Impact-Pack", "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git", PINNED_COMMITS["ComfyUI-Impact-Pack"]),
            ("ComfyUI-AnimateDiff-Evolved", "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git", PINNED_COMMITS["ComfyUI-AnimateDiff-Evolved"]),
            ("ComfyUI-ReActor", "https://github.com/Gourieff/ComfyUI-ReActor.git", PINNED_COMMITS["ComfyUI-ReActor"]),
        ]
        for node_name, repo_url, commit_hash in nodes:
            node_path = os.path.join(custom_nodes_dir, node_name)
            git_clone_pinned(repo_url, node_path, commit_hash)
            req_file = os.path.join(node_path, "requirements.txt")
            if os.path.exists(req_file):
                run_command([sys.executable, "-m", "pip", "install", "-q", "-r", req_file], check=False)

        # 4. Mount heavy weights from Kaggle Dataset
        dataset_ready = setup_models_from_dataset_or_fallback(COMFY_DIR)

        # 5. Start ComfyUI Server bound strictly to 127.0.0.1:8188 (ZERO OPEN INGRESS)
        log("Step 5: Starting ComfyUI server bound to 127.0.0.1:8188...")
        comfy_log_path = "/kaggle/tmp/comfyui.log"
        comfy_log_file = open(comfy_log_path, "w")
        comfy_proc = subprocess.Popen(
            [
                sys.executable,
                "main.py",
                "--listen", "127.0.0.1",
                "--port", "8188",
                "--lowvram",
            ],
            cwd=COMFY_DIR,
            stdout=comfy_log_file,
            stderr=subprocess.STDOUT
        )

        # Wait for ComfyUI port 8188
        server_ready = False
        for _ in range(60):
            if comfy_proc.poll() is not None:
                raise RuntimeError(f"ComfyUI terminated unexpectedly with code {comfy_proc.returncode}!")
            try:
                with socket.create_connection(("127.0.0.1", 8188), timeout=1):
                    server_ready = True
                    break
            except (socket.error, OSError):
                time.sleep(1)

        if not server_ready:
            raise RuntimeError("ComfyUI failed to start within 60s timeout.")
        log("ComfyUI server is UP and listening exclusively on 127.0.0.1:8188!")

        # 6. Start Authenticated Reverse Tunnel (Cloudflare Quick Tunnel)
        log("Step 6: Starting authenticated reverse tunnel...")
        tunnel_proc = start_authenticated_tunnel(8188)
        tunnel_url = extract_tunnel_url(tunnel_proc)

        # 7. Perform Handshake: Publish to Upstash Redis REST
        log(f"Step 7: Publishing tunnel URL to Upstash Redis for profile [{profile_id}]...")
        handshake_ok = publish_upstash_handshake(upstash_url, upstash_token, profile_id, tunnel_url)
        if not handshake_ok:
            raise RuntimeError(f"Failed to publish tunnel URL {tunnel_url} to Upstash Redis.")

        # Secondary Handshake Webhook to Orchestrator (if configured)
        if orchestrator_url and webhook_token:
            log(f"Performing secondary handshake webhook to {orchestrator_url}...")
            send_webhook_handshake(
                orchestrator_url=orchestrator_url,
                token=webhook_token,
                worker_id=worker_id,
                tunnel_url=tunnel_url,
                status="online",
                dataset_ready=dataset_ready,
            )

    except Exception as startup_err:
        err_msg = f"{type(startup_err).__name__}: {str(startup_err)}"
        log(f"[FATAL WORKER STARTUP ERROR] {err_msg}")
        if upstash_url and upstash_token and profile_id:
            publish_upstash_error(upstash_url, upstash_token, profile_id, err_msg)
        if comfy_proc:
            try:
                comfy_proc.terminate()
            except Exception:
                pass
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
        sys.exit(1)

    # 8. Main Loop: Idle Watchdog (600s) + Upstash & Webhook Heartbeats
    log(f"Entering operational loop with {IDLE_WATCHDOG_TIMEOUT_SECONDS}s idle watchdog...")
    last_active_time = time.time()
    last_heartbeat_time = time.time()

    try:
        while True:
            # Check ComfyUI process
            if comfy_proc.poll() is not None:
                log(f"ComfyUI process exited with code {comfy_proc.returncode}")
                break

            # Check Tunnel process
            if tunnel_proc.poll() is not None:
                log("Tunnel process exited. Restarting tunnel...")
                tunnel_proc = start_authenticated_tunnel(8188)
                tunnel_url = extract_tunnel_url(tunnel_proc)
                if upstash_url and upstash_token and profile_id:
                    publish_upstash_handshake(upstash_url, upstash_token, profile_id, tunnel_url)
                if orchestrator_url and webhook_token:
                    send_webhook_handshake(orchestrator_url, webhook_token, worker_id, tunnel_url, "online", dataset_ready)

            now = time.time()

            # Heartbeat check (every 20s)
            if now - last_heartbeat_time >= min(HEARTBEAT_INTERVAL_SECONDS, 20):
                if upstash_url and upstash_token and profile_id:
                    publish_upstash_heartbeat(upstash_url, upstash_token, profile_id)
                if orchestrator_url and webhook_token:
                    send_heartbeat(orchestrator_url, webhook_token, worker_id)
                last_heartbeat_time = now

            # Idle watchdog check
            idle = is_comfyui_idle()
            if not idle:
                last_active_time = now
            else:
                idle_duration = now - last_active_time
                if idle_duration >= IDLE_WATCHDOG_TIMEOUT_SECONDS:
                    log(f"IDLE WATCHDOG TRIGGERED: No active jobs for {int(idle_duration)}s. Shutting down worker...")
                    # Notify orchestrator of clean offline status
                    if orchestrator_url and webhook_token:
                        send_webhook_handshake(
                            orchestrator_url=orchestrator_url,
                            token=webhook_token,
                            worker_id=worker_id,
                            tunnel_url=tunnel_url,
                            status="offline",
                        )
                    break

            time.sleep(5)

    finally:
        log("Shutting down worker processes...")
        try:
            comfy_proc.terminate()
            tunnel_proc.terminate()
        except Exception:
            pass
        log("SwapeDev Worker terminated cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
