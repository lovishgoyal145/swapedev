#!/usr/bin/env python3
"""
SwapeDev Kaggle Diagnostic & Audit Pipeline
===========================================
Synchronous diagnostic tool to audit:
1. Exact KAGGLE_USERNAME and KAGGLE_KEY resolution (profile vs .env vs host).
2. Exact contents and validity of configs/kernel-metadata.json.
3. Kaggle Python API authentication and kernel status querying.
4. Detailed error capture (including 409 Conflict reason extraction).
5. Comprehensive audit report generation in markdown: kaggle_audit_report.md.
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Ensure swapedev_service is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from swapedev_service.config import (
    get_orchestrator_config,
    get_profile_credentials,
    get_profile_credentials_path,
    has_valid_profile_credentials,
    list_camoufox_profiles,
    get_default_profile_id,
    load_app_env,
)


def resolve_credentials(profile_id: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Resolves Kaggle credentials with clear provenance tracking.
    Returns (username, key, source_description, profile_used).
    """
    config = get_orchestrator_config()
    profiles_dir = config.profiles_dir

    # 1. Explicit profile requested
    if profile_id:
        creds = get_profile_credentials(profile_id, profiles_dir)
        u = creds.get("kaggle_username", "").strip()
        k = creds.get("kaggle_key", "").strip()
        cfg_path = get_profile_credentials_path(profile_id, profiles_dir)
        if u and k:
            return u, k, f"Camoufox profile '{profile_id}' ({cfg_path})", profile_id

    # 2. Check default profile
    def_pid = get_default_profile_id(profiles_dir)
    if def_pid and has_valid_profile_credentials(def_pid, profiles_dir):
        creds = get_profile_credentials(def_pid, profiles_dir)
        u = creds.get("kaggle_username", "").strip()
        k = creds.get("kaggle_key", "").strip()
        if u and k:
            return u, k, f"Default Camoufox profile '{def_pid}'", def_pid

    # 3. Check any profile with valid credentials
    for p in list_camoufox_profiles(profiles_dir):
        pid = p["id"]
        if has_valid_profile_credentials(pid, profiles_dir):
            creds = get_profile_credentials(pid, profiles_dir)
            u = creds.get("kaggle_username", "").strip()
            k = creds.get("kaggle_key", "").strip()
            if u and k:
                return u, k, f"Discovered Camoufox profile '{pid}' with valid credentials", pid

    # 4. Check canonical .env file
    env_vars = load_app_env()
    env_u = (os.getenv("KAGGLE_USERNAME") or env_vars.get("KAGGLE_USERNAME", "")).strip()
    env_k = (os.getenv("KAGGLE_KEY") or env_vars.get("KAGGLE_KEY", "")).strip()
    if env_u and env_k:
        return env_u, env_k, "Environment variables / .env file", ""

    return "", "", "No credentials found in profiles or environment", ""


def get_metadata_path() -> Path:
    """Finds canonical kernel-metadata.json."""
    candidates = [
        Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/configs/kernel-metadata.json"),
        Path.cwd() / "configs" / "kernel-metadata.json",
        Path(__file__).resolve().parent.parent / "configs" / "kernel-metadata.json",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


def run_audit(profile_id: Optional[str] = None, output_report_path: Optional[Path] = None) -> Dict[str, Any]:
    audit_log = []

    def log(msg: str):
        print(msg)
        audit_log.append(msg)

    timestamp = datetime.now().isoformat()
    log("=" * 60)
    log("      SWAPEDEV KAGGLE DIAGNOSTIC AUDIT REPORT")
    log(f"      Execution Timestamp: {timestamp}")
    log("=" * 60)

    # 1. Resolve Credentials
    username, key, source, used_profile = resolve_credentials(profile_id)
    log("\n[1] CREDENTIAL RESOLUTION AUDIT")
    log(f"  * Selected Profile     : {profile_id or used_profile or 'None'}")
    log(f"  * Credential Source    : {source}")
    log(f"  * KAGGLE_USERNAME      : {username or '<EMPTY>'}")
    log(f"  * KAGGLE_KEY           : {key or '<EMPTY>'}")
    log(f"  * Key Type Prefix      : {'KGAT Access Token' if key.startswith('KGAT_') else 'Legacy Key / Custom'}")

    # Check Host Hijacking Risk (~/.kaggle)
    host_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    host_access_token = Path.home() / ".kaggle" / "access_token"
    log("\n[1.1] HOST CREDENTIAL HIJACKING CHECK (~/.kaggle)")
    if host_access_token.exists():
        token_preview = host_access_token.read_text().strip()
        log(f"  [!] WARNING: Host access_token exists: {host_access_token}")
        log(f"      Token value: {token_preview[:7]}...{token_preview[-4:]}")
        log("      (Risk: Kaggle Python SDK will default to this token unless KAGGLE_CONFIG_DIR or KAGGLE_API_TOKEN is isolated!)")
    if host_kaggle_json.exists():
        try:
            h_data = json.loads(host_kaggle_json.read_text())
            log(f"  [!] WARNING: Host kaggle.json exists: {host_kaggle_json} (user: {h_data.get('username')})")
        except Exception:
            log(f"  [!] WARNING: Host kaggle.json exists at {host_kaggle_json}")

    # 2. Metadata Audit
    meta_path = get_metadata_path()
    log("\n[2] KERNEL METADATA AUDIT")
    log(f"  * Metadata Path        : {meta_path}")

    meta_content = {}
    if not meta_path.exists():
        log(f"  [!] ERROR: Metadata file does NOT exist at {meta_path}")
    else:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_raw = f.read()
            meta_content = json.loads(meta_raw)
            log("  * Exact Contents:")
            for line in meta_raw.splitlines():
                log(f"      {line}")
        except Exception as e:
            log(f"  [!] ERROR reading metadata: {e}")

    meta_id = meta_content.get("id", "")
    meta_title = meta_content.get("title", "")
    expected_id = f"{username}/swapedev-backend" if username else "<UNKNOWN>/swapedev-backend"

    log("\n[2.1] METADATA INTEGRITY VERIFICATION")
    log(f"  * Expected 'id'        : {expected_id}")
    log(f"  * Actual 'id'          : {meta_id}")
    id_matches = (meta_id == expected_id)
    log(f"  * ID Strict Match      : {'PASSED' if id_matches else 'FAILED'}")

    title_slug = meta_title.lower().replace(" ", "-") if meta_title else ""
    log(f"  * Kernel Title         : '{meta_title}'")
    log(f"  * Title Slug Derived   : '{title_slug}'")
    if meta_id and "/" in meta_id:
        target_slug = meta_id.split("/")[-1]
        if title_slug and title_slug != target_slug:
            log(f"  [!] NOTICE: Title slug '{title_slug}' != metadata id slug '{target_slug}'.")
            log("      If a kernel with this title already exists on Kaggle, SaveKernel may throw 409 Conflict.")

    # 3. Kaggle Python API Status Query
    log("\n[3] KAGGLE PYTHON API QUERY")
    api_success = False
    api_authenticated_user = ""
    kernel_status_result = None
    existing_user_kernels = []
    error_details = None

    if not username or not key:
        log("  [!] Skipping Kaggle API query: Missing username or key.")
    else:
        temp_dir = tempfile.mkdtemp(prefix="kaggle_audit_")
        try:
            # Isolate Kaggle environment to prevent host ~/.kaggle override
            orig_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
            orig_api_token = os.environ.get("KAGGLE_API_TOKEN")
            orig_username = os.environ.get("KAGGLE_USERNAME")
            orig_key = os.environ.get("KAGGLE_KEY")

            os.environ["KAGGLE_CONFIG_DIR"] = temp_dir
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            if key.startswith("KGAT_"):
                os.environ["KAGGLE_API_TOKEN"] = key

            # Write isolated kaggle.json
            cfg_file = os.path.join(temp_dir, "kaggle.json")
            with open(cfg_file, "w", encoding="utf-8") as kf:
                json.dump({"username": username, "key": key}, kf)
            os.chmod(cfg_file, 0o600)

            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                api_authenticated_user = api.get_config_value("username")
                auth_method = api.get_config_value("auth_method")
                log(f"  * API Authenticated As : {api_authenticated_user} (via {auth_method})")
                api_success = True

                # Check if authenticated user matches target username
                if api_authenticated_user != username:
                    log(f"  [!] WARNING: Authenticated user '{api_authenticated_user}' does not match expected '{username}'!")

                # Query status of target kernel
                target_query_id = meta_id or expected_id
                log(f"  * Querying status for  : {target_query_id}")
                try:
                    status_res = api.kernels_status(target_query_id)
                    kernel_status_result = str(status_res)
                    log(f"  * Kernel Status Result : {kernel_status_result}")
                except Exception as status_err:
                    log(f"  * Kernel Status Error  : {status_err}")
                    error_details = str(status_err)

                # List kernels owned by this user
                log(f"  * Querying kernel list for user '{username}'...")
                try:
                    kernels = api.kernels_list(user=username)
                    existing_user_kernels = [k.ref for k in kernels] if kernels else []
                    log(f"  * User Kernels Found   : {existing_user_kernels}")
                except Exception as list_err:
                    log(f"  * Kernel List Error    : {list_err}")

            except Exception as api_init_err:
                log(f"  [!] API Initialization / Auth Error: {api_init_err}")
                error_details = str(api_init_err)

            finally:
                # Restore original environment
                if orig_config_dir:
                    os.environ["KAGGLE_CONFIG_DIR"] = orig_config_dir
                else:
                    os.environ.pop("KAGGLE_CONFIG_DIR", None)
                if orig_api_token:
                    os.environ["KAGGLE_API_TOKEN"] = orig_api_token
                else:
                    os.environ.pop("KAGGLE_API_TOKEN", None)
                if orig_username:
                    os.environ["KAGGLE_USERNAME"] = orig_username
                else:
                    os.environ.pop("KAGGLE_USERNAME", None)
                if orig_key:
                    os.environ["KAGGLE_KEY"] = orig_key
                else:
                    os.environ.pop("KAGGLE_KEY", None)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 4. Diagnostic Assessment & Recommendations
    log("\n[4] DIAGNOSTIC ASSESSMENT & ROOT CAUSE")
    if not id_matches:
        log("  [!] ISSUE: kernel-metadata.json 'id' does not match target username.")
    if host_access_token.exists() and not os.environ.get("KAGGLE_CONFIG_DIR"):
        log("  [!] ISSUE: Host ~/.kaggle/access_token threatens to hijack Kaggle SDK credentials.")
    if kernel_status_result and ("RUNNING" in kernel_status_result or "QUEUED" in kernel_status_result):
        log(f"  [!] NOTICE: Remote kernel is currently ACTIVE ({kernel_status_result}).")
    log("  [✓] Audit execution complete.")
    log("=" * 60)

    # 5. Write Report to Markdown File
    report_content = "\n".join(audit_log)
    if not output_report_path:
        output_report_path = Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/kaggle_audit_report.md")

    try:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as rf:
            rf.write("# Kaggle Diagnostic Audit Report\n\n```text\n")
            rf.write(report_content)
            rf.write("\n```\n")
        print(f"\nAudit report successfully generated: {output_report_path}")

        # Also write a local copy if running from another directory
        local_report = Path.cwd() / "kaggle_audit_report.md"
        if local_report.resolve() != output_report_path.resolve():
            with open(local_report, "w", encoding="utf-8") as rf:
                rf.write("# Kaggle Diagnostic Audit Report\n\n```text\n")
                rf.write(report_content)
                rf.write("\n```\n")
            print(f"Local copy saved to: {local_report}")
    except Exception as save_err:
        print(f"Warning: Could not save report file: {save_err}")

    return {
        "timestamp": timestamp,
        "username": username,
        "key": key,
        "source": source,
        "metadata_path": str(meta_path),
        "metadata_id": meta_id,
        "expected_id": expected_id,
        "id_matches": id_matches,
        "api_authenticated_user": api_authenticated_user,
        "kernel_status": kernel_status_result,
        "existing_user_kernels": existing_user_kernels,
        "error_details": error_details,
    }


def main():
    parser = argparse.ArgumentParser(description="SwapeDev Kaggle Diagnostic Audit Tool")
    parser.add_argument("--profile", "-p", help="Camoufox tenant profile ID to audit (e.g. enigma_drift, maxx)")
    parser.add_argument("--out", "-o", help="Output path for kaggle_audit_report.md")
    args = parser.parse_args()

    out_path = Path(args.out).resolve() if args.out else None
    run_audit(profile_id=args.profile, output_report_path=out_path)


if __name__ == "__main__":
    main()
