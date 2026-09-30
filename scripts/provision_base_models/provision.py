#!/usr/bin/env python3
"""
SwapeDev Base Models Provisioner
================================
One-off Kaggle Kernel script (internet ON) to download the 4 required base models,
verify their exact sizes and SHA-256 hashes, and publish them as the Kaggle dataset
avidok/swapedev-base-models.

Fail-Closed: Any URL failure, size mismatch, or SHA-256 mismatch terminates execution immediately.
"""

import os
import sys
import time
import json
import hashlib
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_SPECS = {
    "sd-v1-5-inpainting.ckpt": {
        "url": "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt",
        "expected_sha256": "c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19",
        "expected_size": 4265437280,
    },
    "sam2_hiera_small.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "expected_sha256": "95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3",
        "expected_size": 184309650,
    },
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "expected_sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
        "expected_size": 554253681,
    },
    "v3_sd15_mm.ckpt": {
        "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
        "expected_sha256": "2412711886f61091846f53204aabc38aa6e09356d62a9808abe4daa802168343",
        "expected_size": 1673262583,
    },
}

EXTRA_FILES = {
    "insightface-0.7.3-cp310-cp310-linux_x86_64.whl": {
        "url": "https://huggingface.co/deauxpas/colabrepo/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl",
        "expected_size": 15315920,
    }
}

DATASET_ID = "avidok/swapedev-base-models"
DATASET_TITLE = "SwapeDev Base Models"
STAGING_DIR = Path("/kaggle/working/swapedev-base-models")


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_secrets() -> dict:
    log("Searching for secrets.json under /kaggle/input...")
    input_p = Path("/kaggle/input")
    if input_p.exists():
        for root, _, files in os.walk(str(input_p)):
            if "secrets.json" in files:
                p = Path(root) / "secrets.json"
                log(f"Discovered secrets at {p}")
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception as e:
                    log(f"Error reading {p}: {e}")
    return {}


def report_upstash_status(up_url: str, up_tok: str, status: str, detail: str, error: str = None):
    if not up_url or not up_tok:
        return
    try:
        base_url = up_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {up_tok}",
            "Content-Type": "application/json",
        }
        payload = json.dumps([
            "SET",
            "swapedev:provisioner:status",
            json.dumps({
                "timestamp": time.time(),
                "status": status,
                "detail": detail,
                "error": error,
            })
        ])
        req = urllib.request.Request(f"{base_url}/", data=payload.encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as e:
        log(f"Warning: Failed to report status to Upstash: {e}")


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, expected_size: int = None) -> bool:
    log(f"Starting download: {dest_path.name} from {url}")
    temp_dest = dest_path.with_suffix(".tmp")
    if temp_dest.exists():
        temp_dest.unlink()

    # Try aria2c if available
    aria2c_bin = shutil.which("aria2c")
    if aria2c_bin:
        log(f"Using aria2c for {dest_path.name}...")
        cmd = [
            aria2c_bin,
            "-c",
            "-x", "8",
            "-s", "8",
            "-k", "1M",
            "-j", "4",
            "-d", str(dest_path.parent),
            "-o", temp_dest.name,
            url,
        ]
        ret = subprocess.run(cmd, check=False)
        if ret.returncode == 0 and temp_dest.exists():
            temp_dest.rename(dest_path)
            log(f"Download complete (aria2c): {dest_path.name} ({dest_path.stat().st_size} bytes)")
            return True
        log("aria2c failed or incomplete, falling back to streaming urllib...")

    # Streaming download via urllib
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    downloaded = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(temp_dest, "wb") as f:
        total_header = resp.headers.get("Content-Length")
        total_len = int(total_header) if total_header else expected_size
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
            pct = (downloaded / total_len * 100) if total_len else 0
            if int(elapsed) % 5 == 0:
                log(f"{dest_path.name}: {downloaded / (1024*1024):.1f} MB ({pct:.1f}%) at {speed_mb:.1f} MB/s")

    temp_dest.rename(dest_path)
    total_time = time.time() - t0
    log(f"Download complete: {dest_path.name} ({dest_path.stat().st_size} bytes in {total_time:.1f}s)")
    return True


def main():
    log("=== Starting SwapeDev Base Models Provisioner ===")
    secrets = load_secrets()
    k_user = secrets.get("KAGGLE_USERNAME") or os.getenv("KAGGLE_USERNAME")
    k_key = secrets.get("KAGGLE_KEY") or os.getenv("KAGGLE_KEY")
    up_url = secrets.get("UPSTASH_REDIS_REST_URL") or os.getenv("UPSTASH_REDIS_REST_URL")
    up_tok = secrets.get("UPSTASH_REDIS_REST_TOKEN") or os.getenv("UPSTASH_REDIS_REST_TOKEN")

    if not k_user or not k_key:
        err = "FATAL: Missing KAGGLE_USERNAME or KAGGLE_KEY in secrets or env!"
        log(err)
        report_upstash_status(up_url, up_tok, "FAILED", err, error=err)
        sys.exit(1)

    log(f"Authenticated as Kaggle user: {k_user}")
    report_upstash_status(up_url, up_tok, "RUNNING", "Provisioner started on Kaggle")

    # Configure Kaggle credentials for SDK / CLI
    os.environ["KAGGLE_USERNAME"] = k_user
    os.environ["KAGGLE_KEY"] = k_key
    if k_key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = k_key

    dot_kaggle = Path(os.path.expanduser("~/.kaggle"))
    dot_kaggle.mkdir(parents=True, exist_ok=True)
    (dot_kaggle / "kaggle.json").write_text(json.dumps({"username": k_user, "key": k_key}), encoding="utf-8")
    os.chmod(dot_kaggle / "kaggle.json", 0o600)
    (dot_kaggle / "access_token").write_text(k_key, encoding="utf-8")
    os.chmod(dot_kaggle / "access_token", 0o600)

    # Prepare staging directory
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = STAGING_DIR / "dataset-metadata.json"
    meta_path.write_text(json.dumps({
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}]
    }, indent=2), encoding="utf-8")

    # Download and verify each model
    for filename, spec in MODEL_SPECS.items():
        dest = STAGING_DIR / filename
        log(f"--- Processing {filename} ---")
        report_upstash_status(up_url, up_tok, "DOWNLOADING", f"Downloading {filename}")
        
        # Download
        success = download_file(spec["url"], dest, spec["expected_size"])
        if not success or not dest.exists():
            err = f"FATAL: Failed to download {filename}"
            log(err)
            report_upstash_status(up_url, up_tok, "FAILED", err, error=err)
            sys.exit(1)

        # Check size
        actual_size = dest.stat().st_size
        if actual_size != spec["expected_size"]:
            err = f"FATAL: Size mismatch for {filename}: expected {spec['expected_size']}, got {actual_size}"
            log(err)
            report_upstash_status(up_url, up_tok, "FAILED", err, error=err)
            sys.exit(1)
        log(f"Size verified: {actual_size} bytes")

        # Check SHA-256
        report_upstash_status(up_url, up_tok, "VERIFYING", f"Verifying SHA-256 for {filename}")
        log(f"Computing SHA-256 for {filename}...")
        actual_sha = compute_sha256(dest)
        if actual_sha != spec["expected_sha256"]:
            err = f"FATAL: SHA-256 mismatch for {filename}: expected {spec['expected_sha256']}, got {actual_sha}"
            log(err)
            report_upstash_status(up_url, up_tok, "FAILED", err, error=err)
            sys.exit(1)
        log(f"SHA-256 PASS: {actual_sha}")

    # Download extra files (insightface wheel)
    for filename, extra in EXTRA_FILES.items():
        dest = STAGING_DIR / filename
        log(f"--- Downloading extra file {filename} ---")
        download_file(extra["url"], dest, extra.get("expected_size"))

    # Publish dataset to Kaggle
    log("--- Publishing dataset to Kaggle ---")
    report_upstash_status(up_url, up_tok, "UPLOADING", f"Uploading {DATASET_ID} to Kaggle")

    # Try Kaggle Python API first
    published = False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Check if already exists
        existing = False
        try:
            mine = api.dataset_list(mine=True)
            for d in mine:
                if d and d.ref and d.ref.lower() == DATASET_ID.lower():
                    existing = True
                    break
        except Exception as e:
            log(f"Error checking existing datasets: {e}")

        if not existing:
            log(f"Creating new private dataset: {DATASET_ID}")
            res = api.dataset_create_new(folder=str(STAGING_DIR), public=False, quiet=False, dir_mode="skip")
            log(f"dataset_create_new result: {res}")
            published = True
        else:
            log(f"Updating existing dataset: {DATASET_ID}")
            res = api.dataset_create_version(folder=str(STAGING_DIR), version_notes="SwapeDev base models initial release", quiet=False, dir_mode="skip")
            log(f"dataset_create_version result: {res}")
            published = True
    except Exception as e:
        log(f"Kaggle SDK publish failed: {e}. Trying CLI fallback...")

    if not published:
        # CLI fallback
        log("Running CLI: kaggle datasets create...")
        res = subprocess.run(["kaggle", "datasets", "create", "-p", str(STAGING_DIR), "-r", "skip"], capture_output=True, text=True)
        log(f"CLI create STDOUT: {res.stdout}")
        log(f"CLI create STDERR: {res.stderr}")
        if res.returncode != 0:
            log("CLI create failed, trying kaggle datasets version...")
            res2 = subprocess.run(["kaggle", "datasets", "version", "-p", str(STAGING_DIR), "-m", "Initial release", "-r", "skip"], capture_output=True, text=True)
            log(f"CLI version STDOUT: {res2.stdout}")
            log(f"CLI version STDERR: {res2.stderr}")
            if res2.returncode != 0:
                err = f"FATAL: Failed to publish dataset via both SDK and CLI!"
                log(err)
                report_upstash_status(up_url, up_tok, "FAILED", err, error=err)
                sys.exit(1)

    log("=== Provisioning Complete! Dataset published successfully ===")
    report_upstash_status(up_url, up_tok, "COMPLETE", f"Dataset {DATASET_ID} successfully created and verified")
    sys.exit(0)


if __name__ == "__main__":
    main()
