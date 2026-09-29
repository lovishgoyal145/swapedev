"""
Test Suite for Strict Per-Profile Compute Isolation
===================================================
Validates:
1. Profile-Scoped Storage: Credentials saved to profiles/{profile_id}/config.json with chmod 0600.
2. Gatekeeper Tenancy Enforcement: Redirects to /setup?profile_id={pid} if any of the 4 credentials
   (KAGGLE_USERNAME, KAGGLE_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN) are missing.
3. Zero Cross-Profile Leakage: Configured Profile A never allows unconfigured Profile B to bypass Gatekeeper.
4. Setup UI: Contains inputs for all 4 credentials and saves them to profile config.
5. KaggleApi Secrets: set_user_secret() invoked for UPSTASH_REST_URL, UPSTASH_REST_TOKEN, and PROFILE_ID.
6. Zero Global os.environ Pollution: Server environment is never polluted with profile credentials.
7. Upstash Namespacing & State Cleanliness: Handshake key swapedev:{profile_id}:tunnel_url is polled
   and immediately deleted upon retrieval.
8. Zero Quota Bleed: Operations on Profile A never alter, poll, or cancel Profile B\x27s compute resources.
"""

import os
import stat
import json
import time
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from swapedev_service.ui_app import app as ui_app
from swapedev_service.config import (
    get_orchestrator_config,
    get_profile_credentials,
    get_profile_credentials_path,
    has_valid_profile_credentials,
    save_profile_credentials,
    get_profile_credential_status,
)
from swapedev_service.worker_manager import (
    WorkerManager,
    WorkerState,
    get_worker_manager,
    query_upstash_command,
    delete_upstash_key,
)


@pytest.fixture(autouse=True)
def setup_test_profiles_env():
    """Isolates profiles directory and restores state before/after each test."""
    config = get_orchestrator_config()
    profiles_dir = config.profiles_dir
    wm = get_worker_manager()

    def cleanup():
        for p in profiles_dir.iterdir():
            if p.is_dir() and (p.name in ("profile_maxx", "maxx", "profile_enigma_drift", "enigma_drift", "profile_alpha", "profile_beta") or p.name.startswith(("test_", "profile_test_"))):
                s_dir = p / "swapedev"
                if s_dir.exists():
                    shutil.rmtree(s_dir, ignore_errors=True)
                root_cfg = p / "config.json"
                if root_cfg.exists():
                    root_cfg.unlink(missing_ok=True)
        wm._states.clear()
        wm._boot_tasks.clear()
        wm._timeout_tasks.clear()
        wm._poll_tasks.clear()
        if wm.state_file.exists():
            wm.state_file.unlink(missing_ok=True)
        if wm.workers_dir.exists():
            for f in wm.workers_dir.glob("*.json"):
                f.unlink(missing_ok=True)

    cleanup()
    yield
    cleanup()


def test_profile_scoped_credentials_storage_and_permissions():
    """Verify credentials storage strictly in profiles/{profile_id}/config.json with chmod 0600."""
    pid = "profile_maxx"
    config = get_orchestrator_config()
    profiles_dir = config.profiles_dir

    saved_path = save_profile_credentials(
        profile_id=pid,
        username="kaggle_maxx_user",
        key="kaggle_maxx_secret_key_123",
        upstash_redis_rest_url="https://maxx-isolated-db.upstash.io",
        upstash_redis_rest_token="maxx_token_secret_xyz",
    )

    assert saved_path.exists()
    assert str(saved_path).endswith("profiles/profile_maxx/config.json")

    # Verify chmod 0600
    file_mode = stat.S_IMODE(saved_path.stat().st_mode)
    assert file_mode == 0o600, f"Expected 0600 permissions, got {oct(file_mode)}"

    # Verify content
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["KAGGLE_USERNAME"] == "kaggle_maxx_user"
    assert data["KAGGLE_KEY"] == "kaggle_maxx_secret_key_123"
    assert data["UPSTASH_REDIS_REST_URL"] == "https://maxx-isolated-db.upstash.io"
    assert data["UPSTASH_REDIS_REST_TOKEN"] == "maxx_token_secret_xyz"

    # Verify get_profile_credentials
    creds = get_profile_credentials(pid, profiles_dir)
    assert creds["kaggle_username"] == "kaggle_maxx_user"
    assert creds["kaggle_key"] == "kaggle_maxx_secret_key_123"
    assert creds["upstash_redis_rest_url"] == "https://maxx-isolated-db.upstash.io"
    assert creds["upstash_redis_rest_token"] == "maxx_token_secret_xyz"

    # Verify validation
    assert has_valid_profile_credentials(pid, profiles_dir) is True


def test_has_valid_profile_credentials_requires_all_four_fields():
    """Verify all 4 fields are strictly required; missing any field fails validation."""
    pid = "profile_enigma_drift"
    config = get_orchestrator_config()
    profiles_dir = config.profiles_dir

    # Incomplete: missing Upstash credentials
    save_profile_credentials(
        profile_id=pid,
        username="enigma_user",
        key="enigma_key",
        upstash_redis_rest_url="",
        upstash_redis_rest_token="",
    )
    assert has_valid_profile_credentials(pid, profiles_dir) is False

    # Incomplete: missing Kaggle key
    save_profile_credentials(
        profile_id=pid,
        username="enigma_user",
        key="",
        upstash_redis_rest_url="https://enigma-db.upstash.io",
        upstash_redis_rest_token="token_enigma",
    )
    assert has_valid_profile_credentials(pid, profiles_dir) is False

    # Complete: all 4 fields valid
    save_profile_credentials(
        profile_id=pid,
        username="enigma_user",
        key="enigma_key",
        upstash_redis_rest_url="https://enigma-db.upstash.io",
        upstash_redis_rest_token="token_enigma",
    )
    assert has_valid_profile_credentials(pid, profiles_dir) is True


def test_gatekeeper_profile_isolation_and_redirection():
    """Verify Gatekeeper redirects unconfigured profile without leaking configured profile."""
    client = TestClient(ui_app, follow_redirects=False)

    # Configure profile_maxx
    save_profile_credentials(
        profile_id="profile_maxx",
        username="user_maxx",
        key="key_maxx",
        upstash_redis_rest_url="https://maxx.upstash.io",
        upstash_redis_rest_token="tok_maxx",
    )

    # profile_enigma_drift is unconfigured
    res_enigma = client.get("/dashboard?profile_id=profile_enigma_drift")
    assert res_enigma.status_code == 307
    assert "/setup?profile_id=" in res_enigma.headers.get("location")
    assert "enigma_drift" in res_enigma.headers.get("location")

    # profile_maxx is fully configured and allowed access
    res_maxx = client.get("/dashboard?profile_id=profile_maxx")
    assert res_maxx.status_code == 200
    assert "SwapeDev" in res_maxx.text


def test_setup_form_contains_all_four_credential_inputs_and_saves():
    """Verify setup form displays all 4 fields and saves them properly."""
    client = TestClient(ui_app, follow_redirects=False)

    # GET setup page
    res_get = client.get("/setup?profile_id=profile_enigma_drift")
    assert res_get.status_code == 200
    html = res_get.text
    assert 'name="username"' in html
    assert 'name="key"' in html
    assert 'name="upstash_url"' in html
    assert 'name="upstash_token"' in html

    # POST setup page
    post_data = {
        "profile_id": "profile_enigma_drift",
        "username": "new_enigma_user",
        "key": "new_enigma_key_999",
        "upstash_url": "https://new-enigma.upstash.io",
        "upstash_token": "token_new_999",
    }
    res_post = client.post("/setup", data=post_data)
    assert res_post.status_code == 303
    assert "/dashboard?profile_id=" in res_post.headers.get("location")
    assert "enigma_drift" in res_post.headers.get("location")

    # Verify credentials saved with chmod 0600
    config = get_orchestrator_config()
    creds = get_profile_credentials("profile_enigma_drift", config.profiles_dir)
    assert creds["kaggle_username"] == "new_enigma_user"
    assert creds["kaggle_key"] == "new_enigma_key_999"
    assert creds["upstash_redis_rest_url"] == "https://new-enigma.upstash.io"
    assert creds["upstash_redis_rest_token"] == "token_new_999"

    cfg_file = get_profile_credentials_path("profile_enigma_drift", config.profiles_dir)
    assert stat.S_IMODE(cfg_file.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_zero_global_os_environ_pollution():
    """Verify os.environ in orchestrator process is NEVER mutated with Kaggle/Upstash credentials."""
    client = TestClient(ui_app)

    # Snapshot global os.environ
    saved_username = os.environ.get("KAGGLE_USERNAME")
    saved_key = os.environ.get("KAGGLE_KEY")
    saved_conf_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    saved_token = os.environ.get("KAGGLE_API_TOKEN")

    save_profile_credentials(
        profile_id="maxx",
        username="isolated_worker_user",
        key="isolated_worker_secret_key",
        upstash_redis_rest_url="https://maxx-db.upstash.io",
        upstash_redis_rest_token="maxx_token_secret",
    )

    captured_sub_env = {}

    async def mock_exec(*args, **kwargs):
        captured_sub_env.update(kwargs.get("env", {}))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Kernel successfully pushed", b""))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200

        wm = get_worker_manager()
        boot_task = wm._boot_tasks.get("maxx") or wm._boot_task
        if boot_task:
            await boot_task

        # Verify subprocess received profile credentials
        assert captured_sub_env.get("KAGGLE_USERNAME") == "isolated_worker_user"
        assert captured_sub_env.get("KAGGLE_KEY") == "isolated_worker_secret_key"

        # Verify global os.environ is COMPLETELY UNTOUCHED
        assert os.environ.get("KAGGLE_USERNAME") == saved_username
        assert os.environ.get("KAGGLE_KEY") == saved_key
        assert os.environ.get("KAGGLE_CONFIG_DIR") == saved_conf_dir
        assert os.environ.get("KAGGLE_API_TOKEN") == saved_token


@pytest.mark.asyncio
async def test_kaggle_api_secrets_dataset_sync_for_profile():
    """Verify Option A: private secrets dataset is created/synced and attached to kernel-metadata.json."""
    client = TestClient(ui_app)

    save_profile_credentials(
        profile_id="maxx",
        username="tester_maxx",
        key="test_key_123",
        upstash_redis_rest_url="https://maxx-isolated.upstash.io",
        upstash_redis_rest_token="token_isolated_456",
    )

    mock_exec = AsyncMock()
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"Kernel pushed", b""))
    mock_exec.return_value = mock_proc

    mock_api = MagicMock()
    mock_api.get_config_value.return_value = "tester_maxx"
    mock_api.kernels_status.return_value = "COMPLETE"
    mock_api.dataset_list.return_value = []

    mock_create_resp = MagicMock()
    mock_create_resp.status = "ok"
    mock_create_resp.error = None
    mock_api.dataset_create_new.return_value = mock_create_resp

    with patch("asyncio.create_subprocess_exec", mock_exec), \
         patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=mock_api):

        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200

        wm = get_worker_manager()
        boot_task = wm._boot_tasks.get("maxx") or wm._boot_task
        if boot_task:
            await boot_task

        # Verify dataset_create_new was called with public=False
        assert mock_api.dataset_create_new.called
        kwargs = mock_api.dataset_create_new.call_args[1]
        assert kwargs.get("public") is False

        # Verify kernel metadata has secrets dataset attached to dataset_sources
        meta_path = Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/configs/kernel-metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        assert "tester_maxx/swapedev-secrets-maxx" in meta["dataset_sources"]
        assert "avidok/swapedev-base-models" in meta["dataset_sources"]
        assert meta["is_private"] is True


@pytest.mark.asyncio
async def test_upstash_worker_error_polling():
    """Verify orchestrator detects worker_error from Upstash, sets ERROR state, and cleans key."""
    wm = get_worker_manager()
    pid = "maxx"
    boot_id = "test_err_boot_001"

    st = wm.get_profile_state(pid)
    st.boot_id = boot_id
    st.state = WorkerState.AWAITING_TUNNEL

    test_err = "Fatal: Missing required worker secrets: PROFILE_ID"

    async def mock_query(url, token, *args):
        if len(args) >= 2 and args[1] == f"swapedev:{pid}:worker_error":
            return test_err
        return None

    deleted_keys = []

    async def mock_delete(url, token, key):
        deleted_keys.append(key)
        return True

    with patch("swapedev_service.worker_manager.query_upstash_command", side_effect=mock_query), \
         patch("swapedev_service.worker_manager.delete_upstash_key", side_effect=mock_delete):

        # Run poll loop
        await wm._poll_upstash_tunnel(pid, boot_id, "https://test.upstash.io", "test_token")

        assert st.state == WorkerState.ERROR
        assert "Worker error:" in st.error_message
        assert test_err in st.error_message
        assert f"swapedev:{pid}:worker_error" in deleted_keys


@pytest.mark.asyncio
async def test_upstash_namespacing_and_state_cleanliness():
    """Verify Upstash polling uses swapedev:{profile_id}:tunnel_url and immediately deletes key (State Cleanliness)."""
    wm = get_worker_manager()
    pid = "maxx"
    boot_id = "test_cleanliness_boot_001"

    st = wm.get_profile_state(pid)
    st.boot_id = boot_id
    st.state = WorkerState.AWAITING_TUNNEL

    test_tunnel_url = "https://active-clean-tunnel.trycloudflare.com"

    # Mock query_upstash_command to return the tunnel URL
    async def mock_query(url, token, *args):
        if len(args) >= 2 and args[1] == f"swapedev:{pid}:tunnel_url":
            return test_tunnel_url
        return None

    deleted_keys = []

    async def mock_delete(url, token, key):
        deleted_keys.append(key)
        return True

    with patch("swapedev_service.worker_manager.query_upstash_command", side_effect=mock_query), \
         patch("swapedev_service.worker_manager.delete_upstash_key", side_effect=mock_delete), \
         patch("asyncio.sleep", AsyncMock()):

        await wm._poll_upstash_tunnel(
            profile_id=pid,
            boot_id=boot_id,
            upstash_url="https://real-db.upstash.io",
            upstash_token="valid_token",
        )

        # 1. Verify worker transitioned to READY
        assert st.state == WorkerState.READY
        assert st.tunnel_url == test_tunnel_url
        assert st.connected_at is not None

        # 2. Verify State Cleanliness: key deleted immediately
        assert f"swapedev:{pid}:tunnel_url" in deleted_keys


@pytest.mark.asyncio
async def test_zero_quota_bleed_multi_profile_isolation():
    """Verify operations on profile \x27maxx\x27 never alter, poll, or cancel profile \x27enigma_drift\x27."""
    wm = get_worker_manager()

    save_profile_credentials(
        profile_id="maxx",
        username="user_maxx",
        key="key_maxx",
        upstash_redis_rest_url="https://maxx.upstash.io",
        upstash_redis_rest_token="tok_maxx",
    )
    save_profile_credentials(
        profile_id="enigma_drift",
        username="user_enigma",
        key="key_enigma",
        upstash_redis_rest_url="https://enigma.upstash.io",
        upstash_redis_rest_token="tok_enigma",
    )

    # Set enigma_drift to READY with active tunnel
    st_enigma = wm.get_profile_state("enigma_drift")
    st_enigma.state = WorkerState.READY
    st_enigma.tunnel_url = "https://enigma-tunnel.trycloudflare.com"
    st_enigma.connected_at = time.time()
    wm._save_profile_state("enigma_drift")

    # Set maxx to OFFLINE
    st_maxx = wm.get_profile_state("maxx")
    st_maxx.state = WorkerState.OFFLINE
    wm._save_profile_state("maxx")

    # Stop maxx: ensure enigma_drift is completely unaffected
    with patch("asyncio.create_subprocess_exec") as mock_exec, \
         patch.object(wm, "_cancel_kaggle_kernel", new=AsyncMock()):
        await wm.stop_worker("maxx")

        # Verify maxx is OFFLINE
        assert wm.get_profile_state("maxx").state == WorkerState.OFFLINE

        # Verify enigma_drift remains READY with active tunnel untouched
        enigma_after = wm.get_profile_state("enigma_drift")
        assert enigma_after.state == WorkerState.READY
        assert enigma_after.tunnel_url == "https://enigma-tunnel.trycloudflare.com"

    # Separate state files exist
    assert (wm.workers_dir / "maxx_state.json").exists()
    assert (wm.workers_dir / "enigma_drift_state.json").exists()
