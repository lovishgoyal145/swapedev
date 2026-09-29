"""
Test Suite for Ticket 2: "Warm GPU" Orchestration & Webhook Handshake
=====================================================================
Validates:
1. Initial worker state is OFFLINE.
2. POST /api/worker/start triggers non-blocking Kaggle kernel push.
3. Subprocess inherits KAGGLE_USERNAME and KAGGLE_KEY from .env / profile storage.
4. Subprocess failure transitions state to ERROR.
5. POST /api/worker/webhook with Bearer token transitions state to READY with tunnel URL & uptime.
6. Unauthorized webhook rejects with HTTP 401.
7. 5-minute timeout transitions state to ERROR with "Boot timeout. Kaggle might be queued."
8. POST /api/worker/stop shuts down worker to conserve GPU quota (state OFFLINE).
9. Dashboard HTML integration: Warm GPU button enabled, polling script at 3s interval, shutdown button present.
10. Tenancy invariant: ?profile_id= routing preserved across UI and API endpoints.
"""

import os
import json
import time
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from swapedev_service.ui_app import app as ui_app
from swapedev_service.main import app as main_app
from swapedev_service.config import (
    get_orchestrator_config,
    save_profile_credentials,
)
from swapedev_service.worker_manager import (
    WorkerManager,
    WorkerState,
    get_worker_manager,
)


@pytest.fixture(autouse=True)
def reset_worker_manager():
    """Reset worker manager state before and after each test."""
    config = get_orchestrator_config()
    profiles_dir = config.profiles_dir

    def cleanup():
        for entry in profiles_dir.iterdir():
            if entry.is_dir() and (entry.name in ("profile_maxx", "maxx", "profile_enigma_drift", "enigma_drift") or entry.name.startswith(("test_", "profile_test_"))):
                s_dir = entry / "swapedev"
                if s_dir.exists():
                    shutil.rmtree(s_dir, ignore_errors=True)
                root_cfg = entry / "config.json"
                if root_cfg.exists():
                    root_cfg.unlink(missing_ok=True)

    cleanup()

    # Backup kernel-metadata.json
    meta_path = Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/configs/kernel-metadata.json")
    meta_backup = None
    if meta_path.exists():
        meta_backup = meta_path.read_text(encoding="utf-8")

    # Set up test profile
    save_profile_credentials(
        profile_id="maxx",
        username="tester_maxx",
        key="test_kaggle_key_123",
        upstash_redis_rest_url="https://maxx-db.upstash.io",
        upstash_redis_rest_token="token_maxx_456",
    )

    wm = get_worker_manager()
    # Reset in-memory attributes
    wm._states.clear()
    wm._boot_tasks.clear()
    wm._timeout_tasks.clear()
    wm._poll_tasks.clear()
    if wm.state_file.exists():
        wm.state_file.unlink(missing_ok=True)
    if wm.workers_dir.exists():
        for f in wm.workers_dir.glob("*.json"):
            f.unlink(missing_ok=True)

    yield wm

    # Teardown
    wm._states.clear()
    wm._boot_tasks.clear()
    wm._timeout_tasks.clear()
    wm._poll_tasks.clear()
    if wm.state_file.exists():
        wm.state_file.unlink(missing_ok=True)
    if wm.workers_dir.exists():
        for f in wm.workers_dir.glob("*.json"):
            f.unlink(missing_ok=True)

    # Restore kernel-metadata.json
    if meta_backup and meta_path.exists():
        meta_path.write_text(meta_backup, encoding="utf-8")

    cleanup()


def test_initial_worker_status_offline():
    client = TestClient(ui_app)
    res = client.get("/api/worker/status")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "OFFLINE"
    assert data["connected"] is False
    assert data["tunnel_url"] is None
    assert data["uptime"] == 0.0
    assert data["error"] is None


@pytest.mark.asyncio
async def test_worker_start_non_blocking_and_env_inheritance():
    client = TestClient(ui_app)

    captured_cmd = []
    captured_env = {}

    async def mock_exec(*args, **kwargs):
        captured_cmd.extend(args)
        captured_env.update(kwargs.get("env", {}))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Kernel successfully pushed", b""))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("started", "already_booting")
        assert data["state"] in ("BOOTING", "AWAITING_TUNNEL")

        # Give event loop a cycle for the background task to run
        wm = get_worker_manager()
        boot_task = wm._boot_tasks.get("maxx") or wm._boot_task
        if boot_task:
            await boot_task

        # Verify command and environment
        assert any("kernels" in str(arg) for arg in captured_cmd)
        assert any("push" in str(arg) for arg in captured_cmd)
        assert "KAGGLE_USERNAME" in captured_env
        assert "KAGGLE_KEY" in captured_env
        assert captured_env["KAGGLE_USERNAME"] != ""
        assert captured_env["KAGGLE_KEY"] != ""

        # Verify kernel-metadata.json was mutated strictly to {KAGGLE_USERNAME}/swapedev-backend
        meta_path = Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/configs/kernel-metadata.json")
        assert meta_path.exists()
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta_data["id"] == f"{captured_env['KAGGLE_USERNAME']}/swapedev-backend"

        # Status should now be AWAITING_TUNNEL
        res_status = client.get("/api/worker/status?profile_id=maxx")
        assert res_status.json()["state"] == "AWAITING_TUNNEL"


@pytest.mark.asyncio
async def test_worker_start_mutates_kernel_metadata_id_for_profile():
    client = TestClient(ui_app)

    # Save credentials for enigma_drift
    save_profile_credentials(
        profile_id="enigma_drift",
        username="enigmad",
        key="KGAT_test_key_xyz",
        upstash_redis_rest_url="https://enigma-db.upstash.io",
        upstash_redis_rest_token="token_enigma_xyz",
    )

    async def mock_exec(*args, **kwargs):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Kernel successfully pushed", b""))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        res = client.post("/api/worker/start?profile_id=enigma_drift")
        assert res.status_code == 200

        wm = get_worker_manager()
        if wm._boot_task:
            await wm._boot_task

        meta_path = Path("/home/lovish/.gemini/antigravity/scratch/SwapeDev/configs/kernel-metadata.json")
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta_data["id"] == "enigmad/swapedev-backend"


@pytest.mark.asyncio
async def test_worker_start_failure_transitions_to_error():
    client = TestClient(ui_app)

    async def mock_exec_fail(*args, **kwargs):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error 403: Forbidden - Out of weekly GPU quota"))
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec_fail):
        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200

        wm = get_worker_manager()
        if wm._boot_task:
            await wm._boot_task

        res_status = client.get("/api/worker/status")
        data = res_status.json()
        assert data["state"] == "ERROR"
        assert "Out of weekly GPU quota" in data["error"]


def test_webhook_handshake_transitions_to_ready():
    client = TestClient(ui_app)
    config = get_orchestrator_config()
    token = config.webhook_token

    payload = {
        "worker_id": "kaggle_t4_test_worker_01",
        "tunnel_url": "https://fast-tunnel.trycloudflare.com",
        "status": "online",
        "gpu_info": "Tesla T4, 15360 MiB",
        "dataset_ready": True,
        "timestamp": time.time(),
    }

    # 1. Successful authenticated handshake
    headers = {"Authorization": f"Bearer {token}"}
    res_webhook = client.post("/api/worker/webhook", json=payload, headers=headers)
    assert res_webhook.status_code == 200
    assert res_webhook.json()["status"] == "accepted"
    assert res_webhook.json()["tunnel_url"] == "https://fast-tunnel.trycloudflare.com"

    # 2. Status verifies READY
    res_status = client.get("/api/worker/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["state"] == "READY"
    assert status_data["connected"] is True
    assert status_data["tunnel_url"] == "https://fast-tunnel.trycloudflare.com"
    assert status_data["worker_id"] == "kaggle_t4_test_worker_01"


def test_webhook_unauthorized_token():
    client = TestClient(ui_app)
    payload = {
        "worker_id": "kaggle_t4_rogue",
        "tunnel_url": "https://rogue-tunnel.trycloudflare.com",
        "status": "online",
    }
    # Bad token
    headers = {"Authorization": "Bearer invalid_token_xyz"}
    res = client.post("/api/worker/webhook", json=payload, headers=headers)
    assert res.status_code == 401

    # Missing header
    res_no_auth = client.post("/api/worker/webhook", json=payload)
    assert res_no_auth.status_code == 401


def test_5_minute_timeout_watchdog():
    client = TestClient(ui_app)
    wm = get_worker_manager()

    # Simulate worker booting with start time > 300s ago
    wm.state = WorkerState.AWAITING_TUNNEL
    wm.boot_started_at = time.time() - 305  # 5 minutes and 5 seconds ago
    wm._save_state()

    # Status query triggers deterministic timeout evaluation
    res = client.get("/api/worker/status")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "ERROR"
    assert data["error"] == "Boot timeout. Kaggle might be queued."


@pytest.mark.asyncio
async def test_worker_stop_quota_saver():
    client = TestClient(ui_app)
    config = get_orchestrator_config()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"Kernel status: Stopped", b""))
        mock_exec.return_value = mock_proc

        # Set worker to READY first
        client.post(
            "/api/worker/webhook",
            json={"worker_id": "t4_worker", "tunnel_url": "https://active-tunnel.com", "status": "online"},
            headers={"Authorization": f"Bearer {config.webhook_token}"},
        )
        assert client.get("/api/worker/status").json()["state"] == "READY"

        # Call stop
        res_stop = client.post("/api/worker/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["state"] == "OFFLINE"

        # Status is now OFFLINE
        res_status = client.get("/api/worker/status")
        assert res_status.json()["state"] == "OFFLINE"
        assert res_status.json()["connected"] is False
        assert res_status.json()["tunnel_url"] is None


def test_dashboard_ui_elements_and_polling():
    client = TestClient(ui_app)
    res = client.get("/dashboard?profile_id=maxx")
    assert res.status_code == 200
    html = res.text

    # 1. Warm GPU button is enabled
    assert 'id="warmGpuBtn"' in html
    assert "Warm GPU (Boot T4 Worker)" in html
    # Check that warmGpuBtn does NOT have disabled attribute
    assert 'id="warmGpuBtn"\n            type="button"\n            disabled' not in html

    # 2. Shutdown button exists
    assert 'id="shutdownGpuBtn"' in html
    assert "Shutdown GPU" in html

    # 3. Alert banner exists with boot timeout text
    assert 'id="workerAlertBanner"' in html
    assert "Boot timeout. Kaggle might be queued." in html

    # 4. Polling script running every 3000ms
    assert "fetchWorkerStatus" in html or "pollStatus" in html
    assert "3000" in html
    assert "/api/worker/status" in html
    assert "/api/worker/start" in html
    assert "/api/worker/stop" in html


def test_tenancy_routing_invariant():
    """Verify ?profile_id= remains intact and API endpoints bypass gatekeeper redirect."""
    client = TestClient(ui_app, follow_redirects=False)

    # API endpoints return 200 JSON, not 307 redirect
    res_status = client.get("/api/worker/status?profile_id=maxx")
    assert res_status.status_code == 200
    assert res_status.headers.get("content-type", "").startswith("application/json")

    # Root auto-routes with profile
    res_root = client.get("/?profile_id=maxx")
    assert res_root.status_code == 303
    assert "/dashboard?profile_id=maxx" in res_root.headers.get("location")


@pytest.mark.asyncio
async def test_preflight_active_kernel_status_check():
    """Validates Step B preflight: graceful attachment when kernel is RUNNING (no push, no cancel)."""
    client = TestClient(ui_app)

    mock_exec = AsyncMock()

    mock_api = MagicMock()
    mock_api.get_config_value.return_value = "tester_maxx"
    mock_api.kernels_status.return_value = "RUNNING"
    mock_api.kernels_logs.return_value = "Initializing SwapeDev environment..."

    with patch("asyncio.create_subprocess_exec", mock_exec), \
         patch("asyncio.sleep", AsyncMock()), \
         patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=mock_api):
        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200

        wm = get_worker_manager()
        if wm._boot_task:
            await wm._boot_task

        # Verify kernels_status was checked
        mock_api.kernels_status.assert_called()
        # Verify NO push was executed (create_subprocess_exec not called)
        mock_exec.assert_not_called()
        # Verify state transitioned gracefully to AWAITING_TUNNEL
        assert wm.state == WorkerState.AWAITING_TUNNEL
        assert wm.tunnel_url is None


@pytest.mark.asyncio
async def test_preflight_active_kernel_graceful_attachment_url_recovery():
    """Validates Step B URL recovery: extracts trycloudflare.com from logs and transitions to READY."""
    client = TestClient(ui_app)

    mock_exec = AsyncMock()

    mock_api = MagicMock()
    mock_api.get_config_value.return_value = "tester_maxx"
    mock_api.kernels_status.return_value = "RUNNING"
    mock_api.kernels_logs.return_value = (
        "2026-09-28 14:00:00 [INFO] Cloudflare tunnel started.\n"
        "2026-09-28 14:00:05 [INFO] Your quick Tunnel has been created! Visit it at: "
        "https://recovered-worker-99.trycloudflare.com\n"
        "2026-09-28 14:00:10 [INFO] FastApi app running on port 8000"
    )

    with patch("asyncio.create_subprocess_exec", mock_exec), \
         patch("asyncio.sleep", AsyncMock()), \
         patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=mock_api):
        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200

        wm = get_worker_manager()
        if wm._boot_task:
            await wm._boot_task

        # Verify no destructive push
        mock_exec.assert_not_called()
        # Verify state is READY and tunnel URL is recovered
        assert wm.state == WorkerState.READY
        assert wm.tunnel_url == "https://recovered-worker-99.trycloudflare.com"

        # Verify GET /api/worker/status reflects READY with recovered tunnel URL
        res_status = client.get("/api/worker/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["state"] == "READY"
        assert status_data["connected"] is True
        assert status_data["tunnel_url"] == "https://recovered-worker-99.trycloudflare.com"



@pytest.mark.asyncio
async def test_verbose_409_conflict_error_body_capture():
    """Validates Step D verbose error capture extracting 409 response details."""
    client = TestClient(ui_app)

    async def mock_exec_409(*args, **kwargs):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(
            b"",
            b"409 Client Error: Conflict for url: https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
        ))
        return mock_proc

    mock_api = MagicMock()
    mock_api.get_config_value.return_value = "tester_maxx"
    mock_api.kernels_status.return_value = "COMPLETE"

    # Mock direct save_kernel to simulate Kaggle returning 409 with JSON body
    import requests
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.text = '{"error":{"code":409,"message":"The requested title is already in use by a notebook. Please choose another title.","status":"ALREADY_EXISTS"}}'
    mock_resp.json.return_value = {
        "error": {
            "code": 409,
            "message": "The requested title is already in use by a notebook. Please choose another title.",
            "status": "ALREADY_EXISTS"
        }
    }
    http_error = requests.exceptions.HTTPError(response=mock_resp)

    mock_client = MagicMock()
    mock_client.kernels.kernels_api_client.save_kernel.side_effect = http_error
    mock_api.build_kaggle_client.return_value.__enter__.return_value = mock_client

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec_409), \
         patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=mock_api):
        res = client.post("/api/worker/start?profile_id=maxx")
        assert res.status_code == 200

        wm = get_worker_manager()
        if wm._boot_task:
            await wm._boot_task

        assert wm.state == WorkerState.ERROR
        assert "The requested title is already in use" in wm.error_message

        # Verify status endpoint returns the exact parsed error
        res_status = client.get("/api/worker/status")
        assert res_status.status_code == 200
        assert "The requested title is already in use" in res_status.json()["error"]


def test_diagnostic_audit_script_synchronous(tmp_path):
    """Validates that diagnostic_audit runs synchronously and creates report."""
    from swapedev_service.diagnostic_audit import run_audit

    report_file = tmp_path / "test_kaggle_audit_report.md"
    result = run_audit(profile_id="maxx", output_report_path=report_file)

    assert result["username"] == "tester_maxx"
    assert result["expected_id"] == "tester_maxx/swapedev-backend"
    assert report_file.exists()
    report_text = report_file.read_text(encoding="utf-8")
    assert "SWAPEDEV KAGGLE DIAGNOSTIC AUDIT REPORT" in report_text
    assert "tester_maxx" in report_text

