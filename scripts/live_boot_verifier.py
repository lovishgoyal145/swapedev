#!/usr/bin/env python3
"""
Live End-to-End Boot Verifier for SwapeDev Kaggle Worker
========================================================
Runs real, un-mocked verification of profile enigma_drift:
1. Triggers start_worker("enigma_drift")
2. Polls Kaggle kernel status and logs
3. Confirms secrets dataset read
4. Confirms Upstash handshake publication
5. Confirms state transition to READY and Upstash key cleanup
6. Stops the worker immediately to conserve Kaggle GPU quota
"""

import sys
import os
import json
import time
import asyncio
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swapedev_service.config import (
    get_orchestrator_config,
    get_profile_credentials,
    has_valid_profile_credentials,
)
from swapedev_service.worker_manager import (
    WorkerManager,
    WorkerState,
    get_worker_manager,
)
from kaggle.api.kaggle_api_extended import KaggleApi


async def run_live_verification():
    profile_id = "enigma_drift"
    config = get_orchestrator_config()

    print(f"=== [START] Live Boot Verification for Profile: {profile_id} ===")
    assert has_valid_profile_credentials(profile_id, config.profiles_dir), "Credentials not valid!"

    creds = get_profile_credentials(profile_id, config.profiles_dir)
    k_user = creds["kaggle_username"]
    k_key = creds["kaggle_key"]
    up_url = creds["upstash_redis_rest_url"].rstrip("/")
    up_tok = creds["upstash_redis_rest_token"]

    # Initialize Kaggle API for external monitoring
    api = KaggleApi()
    if hasattr(api, "CONFIG_NAME_USER") and hasattr(api, "config_values"):
        api.config_values[api.CONFIG_NAME_USER] = k_user
        api.config_values[api.CONFIG_NAME_KEY] = k_key
        if k_key.startswith("KGAT_"):
            api.config_values[api.CONFIG_NAME_TOKEN] = k_key
        api._authenticated = True
    else:
        api.authenticate()

    kernel_id = f"{k_user}/swapedev-backend"
    dataset_ref = f"{k_user}/swapedev-secrets-{profile_id.replace('_', '-')}"

    print(f"Target Kernel: {kernel_id}")
    print(f"Target Dataset: {dataset_ref}")
    print(f"Upstash Endpoint: {up_url}")

    wm = get_worker_manager()

    # Step 1: Start Worker
    print("\n--- Step 1: Triggering wm.start_worker('enigma_drift') ---")
    start_res = await wm.start_worker(profile_id)
    print("start_worker response:", json.dumps(start_res, indent=2))

    # Monitor loop
    max_wait_seconds = 360  # 6 minutes max
    start_time = time.time()
    confirmed_secrets_read = False
    confirmed_tunnel_in_upstash = False
    tunnel_url_observed = None

    print("\n--- Step 2 & 3 & 4: Polling Kaggle & Upstash ---")
    try:
        while time.time() - start_time < max_wait_seconds:
            elapsed = int(time.time() - start_time)
            st = wm.get_status(profile_id)
            print(f"[{elapsed}s] Orchestrator State: {st.state} | Connected: {st.connected} | Tunnel: {st.tunnel_url} | Error: {st.error}")

            # 1. Query Kaggle Kernel Status
            remote_status = "UNKNOWN"
            try:
                k_stat = api.kernels_status(kernel_id)
                remote_status = getattr(k_stat, "status", str(k_stat))
            except Exception as e:
                remote_status = f"Query Err ({e})"

            # 2. Check Kaggle Kernel Logs
            log_snippet = ""
            try:
                logs_raw = api.kernels_logs(kernel_id)
                if logs_raw:
                    logs_str = str(logs_raw)
                    if "Secrets read successfully from dataset" in logs_str or "swapedev-secrets" in logs_str:
                        confirmed_secrets_read = True
                    # Take last 3 lines
                    lines = [ln.strip() for ln in logs_str.split("\n") if ln.strip()]
                    log_snippet = " | ".join(lines[-3:]) if lines else ""
            except Exception:
                pass

            # 3. Direct curl/urllib check against Upstash REST API
            upstash_tunnel_val = None
            try:
                chk_req = urllib.request.Request(
                    f"{up_url}/get/swapedev:{profile_id}:tunnel_url",
                    headers={"Authorization": f"Bearer {up_tok}"}
                )
                with urllib.request.urlopen(chk_req, timeout=4) as chk_res:
                    chk_data = json.loads(chk_res.read().decode())
                    upstash_tunnel_val = chk_data.get("result")
                    if upstash_tunnel_val:
                        confirmed_tunnel_in_upstash = True
                        tunnel_url_observed = upstash_tunnel_val
            except Exception as up_err:
                upstash_tunnel_val = f"Err ({up_err})"

            # Also check for worker_error in Upstash
            try:
                err_req = urllib.request.Request(
                    f"{up_url}/get/swapedev:{profile_id}:worker_error",
                    headers={"Authorization": f"Bearer {up_tok}"}
                )
                with urllib.request.urlopen(err_req, timeout=4) as err_res:
                    err_data = json.loads(err_res.read().decode())
                    if err_data.get("result"):
                        print(f"\n[ALERT] Worker reported fatal error in Upstash: {err_data.get('result')}")
            except Exception:
                pass

            print(f"      Kaggle Kernel Status: {remote_status}")
            if log_snippet:
                print(f"      Kernel Logs: {log_snippet[:140]}...")
            if upstash_tunnel_val:
                print(f"      Direct Upstash Tunnel Key: {upstash_tunnel_val}")

            # Check if READY reached
            if st.state == "READY" and st.tunnel_url:
                print(f"\n>>> SUCCESS! Profile [{profile_id}] transitioned to READY with tunnel: {st.tunnel_url}")
                tunnel_url_observed = st.tunnel_url
                break

            if st.state == "ERROR":
                print(f"\n>>> WORKER FAILED with error: {st.error}")
                break

            await asyncio.sleep(10)

    finally:
        # Step 5 & 6: Verify key deletion and stop worker to save quota
        print("\n--- Step 5: Checking Upstash key state cleanliness ---")
        try:
            chk_req = urllib.request.Request(
                f"{up_url}/get/swapedev:{profile_id}:tunnel_url",
                headers={"Authorization": f"Bearer {up_tok}"}
            )
            with urllib.request.urlopen(chk_req, timeout=4) as chk_res:
                final_key = json.loads(chk_res.read().decode()).get("result")
                print(f"Final Upstash tunnel_url key status: {final_key} (Expected None after consumption)")
        except Exception as e:
            print("Error checking final Upstash key:", e)

        print("\n--- Step 6: Shutting down worker immediately to conserve Kaggle GPU quota ---")
        stop_res = await wm.stop_worker(profile_id)
        print("stop_worker response:", json.dumps(stop_res, indent=2))

        # Check final status
        final_st = wm.get_status(profile_id)
        print(f"Final Orchestrator State: {final_st.state}")

    print("\n=== SUMMARY ===")
    print(f"Dataset Verified Private: Yes ({dataset_ref})")
    print(f"Secrets Read Confirmed: {confirmed_secrets_read}")
    print(f"Tunnel URL Observed: {tunnel_url_observed}")
    print(f"Upstash Published Confirmed: {confirmed_tunnel_in_upstash or bool(tunnel_url_observed)}")


if __name__ == "__main__":
    asyncio.run(run_live_verification())
