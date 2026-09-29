"""
Real Integration Test for Kaggle Worker Dataset Sync & Orchestration
=====================================================================
Exercises the real Kaggle API and secrets dataset creation/verification
when valid credentials are provided in the environment or profile config.

Marked @pytest.mark.integration and @pytest.mark.slow.
Automatically skipped if real KAGGLE_KEY is not configured.
"""

import os
import json
from pathlib import Path
import pytest

from swapedev_service.config import (
    get_orchestrator_config,
    get_profile_credentials,
    has_valid_profile_credentials,
)
from swapedev_service.worker_manager import get_worker_manager


def _get_live_credentials():
    """Retrieves live Kaggle & Upstash credentials if available."""
    config = get_orchestrator_config()
    creds = get_profile_credentials("enigma_drift", config.profiles_dir)

    user = creds.get("kaggle_username") or os.getenv("KAGGLE_USERNAME", "")
    key = creds.get("kaggle_key") or os.getenv("KAGGLE_KEY", "")
    up_url = creds.get("upstash_redis_rest_url") or os.getenv("UPSTASH_REDIS_REST_URL", "")
    up_tok = creds.get("upstash_redis_rest_token") or os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

    # Check for placeholder/test keys
    is_real_key = bool(key and not any(t in key.lower() for t in ("test", "mock", "dummy", "fake")))
    is_real_upstash = bool(up_url and not any(t in up_url.lower() for t in ("mock", "dummy", "fake", "maxx-db", "test-db")))

    if not (user and is_real_key and is_real_upstash and up_tok):
        return None

    return {
        "kaggle_username": user,
        "kaggle_key": key,
        "upstash_url": up_url,
        "upstash_token": up_tok,
    }


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_kaggle_secrets_dataset_sync():
    """
    Live integration test: Verifies Option A Kaggle secrets dataset
    synchronization and privacy check against real Kaggle API.
    """
    creds = _get_live_credentials()
    if not creds:
        pytest.skip("Real Kaggle / Upstash credentials not configured. Skipping live integration test.")

    wm = get_worker_manager()
    pid = "enigma_drift"
    k_user = creds["kaggle_username"]
    k_key = creds["kaggle_key"]
    up_url = creds["upstash_url"]
    up_tok = creds["upstash_token"]

    import tempfile
    from swapedev_service.worker_manager import create_isolated_kaggle_env

    temp_dir = Path(tempfile.mkdtemp(prefix="swapedev_integ_"))
    try:
        create_isolated_kaggle_env(
            username=k_user,
            key=k_key,
            temp_dir=temp_dir,
            profile_id=pid,
            upstash_url=up_url,
            upstash_token=up_tok,
        )

        dataset_ref = await wm._sync_secrets_dataset(
            profile_id=pid,
            kaggle_user=k_user,
            kaggle_key=k_key,
            up_url=up_url,
            up_tok=up_tok,
            temp_config_dir=temp_dir,
        )

        clean_pid = pid.replace("_", "-")
        assert clean_pid in dataset_ref

        # Verify via Kaggle API that the dataset exists and is strictly private
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.config_dir = str(temp_dir)
        if hasattr(api, "CONFIG_NAME_USER") and hasattr(api, "config_values"):
            api.config_values[api.CONFIG_NAME_USER] = k_user
            api.config_values[api.CONFIG_NAME_KEY] = k_key
            if k_key.startswith("KGAT_"):
                api.config_values[api.CONFIG_NAME_TOKEN] = k_key
            api._authenticated = True
        else:
            api.authenticate()

        mine_datasets = api.dataset_list(mine=True)
        matched = [d for d in mine_datasets if d and d.ref and d.ref.lower() == dataset_ref.lower()]
        assert len(matched) > 0, f"Dataset {dataset_ref} not found in user's datasets!"
        assert matched[0].is_private is True, f"Dataset {dataset_ref} is not private!"

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
