"""
Test Suite for Ticket 1: Profile-Isolated Gatekeeper UI
======================================================
Validates:
- Profile Gatekeeper: Unconfigured profile requests routed to /setup?profile_id=...
- Setup form credential submission & secure 0600 profile config.json persistence
- Profile isolation: Credentials for profile A never leak to or satisfy profile B
- Zero profile switch: No "Switch Profile" buttons or links in UI
- Zero local host token sniffing: No ~/.kaggle/kaggle.json read or suggested
- Dashboard admission gating & presence of the 3 required placeholders
"""

import os
import json
import stat
import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from swapedev_service.ui_app import app
from swapedev_service.config import (
    get_orchestrator_config,
    get_profile_swapedev_dir,
    get_profile_credentials_path,
    save_profile_credentials,
    has_valid_profile_credentials,
    list_camoufox_profiles,
)


@pytest.fixture(autouse=True)
def clean_test_swapedev_configs():
    """Ensure profile swapedev test artifacts are cleaned up before and after each test."""
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
    yield
    cleanup()


def test_profile_gatekeeper_unconfigured_redirect():
    client = TestClient(app, follow_redirects=False)

    # 1. Direct dashboard access without credentials redirects to /setup?profile_id=maxx
    res_dash = client.get("/dashboard?profile_id=maxx")
    assert res_dash.status_code in (303, 307)
    assert "/setup?profile_id=maxx" in res_dash.headers.get("location")

    # 2. Setup page renders HTTP 200 with guide and form scoped to tenant
    res_setup = client.get("/setup?profile_id=maxx")
    assert res_setup.status_code == 200
    assert "Configure Profile Credentials" in res_setup.text
    assert "[maxx]" in res_setup.text
    assert 'name="username"' in res_setup.text
    assert 'name="key"' in res_setup.text
    assert 'name="upstash_url"' in res_setup.text
    assert 'name="upstash_token"' in res_setup.text
    assert 'name="profile_id"' in res_setup.text

    # 3. Ensure NO local ~/.kaggle/kaggle.json prefetching or banner
    assert "Local Kaggle Token Detected" not in res_setup.text
    assert "autoFillDetected" not in res_setup.text


def test_setup_form_submission_and_secure_profile_config():
    client = TestClient(app, follow_redirects=False)
    config = get_orchestrator_config()

    # Submit credentials via POST /setup for profile 'maxx'
    res_post = client.post(
        "/setup",
        data={
            "profile_id": "maxx",
            "username": "tester_maxx",
            "key": "test_api_key_maxx_123",
            "upstash_url": "https://oriented-eel-313320.upstash.io",
            "upstash_token": "gQAAAAAABE78AAIgcDJjMGY1NDA1ZGZlMTM0M2NhOGZhODY0NzFhMjQ2OWVlMg",
        },
    )
    assert res_post.status_code == 303
    assert res_post.headers.get("location") == "/dashboard?profile_id=maxx"

    # Verify profile config.json exists with mode 0600
    cfg_path = get_profile_credentials_path("maxx")
    assert cfg_path is not None
    assert cfg_path.exists()
    file_mode = oct(cfg_path.stat().st_mode & 0o777)
    assert file_mode == "0o600"

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["profile_id"] == "maxx"
    assert data["KAGGLE_USERNAME"] == "tester_maxx"
    assert data["KAGGLE_KEY"] == "test_api_key_maxx_123"
    assert data["UPSTASH_REDIS_REST_URL"] == "https://oriented-eel-313320.upstash.io"
    assert data["UPSTASH_REDIS_REST_TOKEN"] == "gQAAAAAABE78AAIgcDJjMGY1NDA1ZGZlMTM0M2NhOGZhODY0NzFhMjQ2OWVlMg"
    assert data["webhook_token"].startswith("swp_sec_")

    # Verify character_vault and staging directories exist inside profile
    s_dir = get_profile_swapedev_dir("maxx")
    assert (s_dir / "character_vault").is_dir()
    assert (s_dir / "staging").is_dir()


def test_tenant_isolation_credentials():
    client = TestClient(app, follow_redirects=False)

    # 1. Configure profile 'maxx' only
    save_profile_credentials(
        profile_id="maxx",
        username="user_maxx",
        key="key_maxx_789",
        upstash_redis_rest_url="https://maxx-db.upstash.io",
        upstash_redis_rest_token="token_maxx_456",
    )
    assert has_valid_profile_credentials("maxx") is True

    # 2. Check profile 'enigma_drift' - must NOT be configured
    assert has_valid_profile_credentials("enigma_drift") is False

    # 3. Accessing 'maxx' dashboard succeeds
    res_maxx = client.get("/dashboard?profile_id=maxx")
    assert res_maxx.status_code == 200
    assert "[maxx]" in res_maxx.text

    # 4. Accessing 'enigma_drift' dashboard redirects to setup for enigma_drift
    res_enigma = client.get("/dashboard?profile_id=enigma_drift")
    assert res_enigma.status_code in (303, 307)
    assert "/setup?profile_id=enigma_drift" in res_enigma.headers.get("location")


def test_zero_profile_switch_in_ui():
    client = TestClient(app, follow_redirects=False)

    save_profile_credentials(
        profile_id="maxx",
        username="user_maxx",
        key="key_maxx_789",
        upstash_redis_rest_url="https://maxx-db.upstash.io",
        upstash_redis_rest_token="token_maxx_456",
    )

    res = client.get("/dashboard?profile_id=maxx")
    assert res.status_code == 200
    html = res.text

    # Verify NO "Switch Profile" button or link
    assert "Switch Profile" not in html
    assert 'title="Switch Profile"' not in html
    assert 'href="/"' not in html

    # Verify Active Tenant context
    assert "SwapeDev Console — Operating as:" in html
    assert "[maxx]" in html

    # Verify Required Placeholders:
    # 1. Warm GPU Button (Ticket 2 active state)
    assert "Warm GPU (Boot T4 Worker)" in html
    assert "disabled" in html

    # 2. Character Library (Identity Vault)
    assert "Character Library (Identity Vault)" in html

    # 3. Video Swap Pipeline
    assert "Video Swap Pipeline" in html


def test_root_auto_routes_to_active_profile():
    client = TestClient(app, follow_redirects=False)

    save_profile_credentials(
        profile_id="maxx",
        username="user_maxx",
        key="key_maxx_789",
        upstash_redis_rest_url="https://maxx-db.upstash.io",
        upstash_redis_rest_token="token_maxx_456",
    )

    # Query param profile_id at root auto-routes directly to dashboard
    res = client.get("/?profile_id=maxx")
    assert res.status_code == 303
    assert res.headers.get("location") == "/dashboard?profile_id=maxx"

    # No selector cards rendered
    assert "Select Profile Context" not in res.text


def test_health_endpoint():
    client = TestClient(app, follow_redirects=False)

    save_profile_credentials(
        profile_id="maxx",
        username="user_maxx",
        key="key_maxx_789",
        upstash_redis_rest_url="https://maxx-db.upstash.io",
        upstash_redis_rest_token="token_maxx_456",
    )

    res = client.get("/health?profile_id=maxx")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8776
    assert data["credentials_configured"] is True
    assert data["profiles_detected_count"] >= 1
