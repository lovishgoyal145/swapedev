"""
SwapeDev Worker API Routes
==========================
Defines the REST API endpoints for Kaggle Worker lifecycle management:
- POST /api/worker/start: Boots remote Kaggle GPU worker (non-blocking)
- GET  /api/worker/status: Returns state, tunnel URL, uptime, error
- POST /api/worker/stop: Cancels kernel to save GPU quota
- POST /api/worker/webhook: Authenticated handshake from Kaggle worker
- POST /api/worker/heartbeat: Keepalive heartbeat
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query, status

from swapedev_service.config import (
    get_orchestrator_config,
    get_profile_credentials,
    list_camoufox_profiles,
)
from swapedev_service.schemas import (
    WorkerHandshakePayload,
    WorkerHeartbeatPayload,
    WorkerStatusResponse,
)
from swapedev_service.worker_manager import (
    WorkerManager,
    WorkerState,
    get_worker_manager,
)

logger = logging.getLogger("swapedev.worker_routes")

router = APIRouter(prefix="/api/worker", tags=["Worker Orchestration"])


def verify_bearer_token(authorization: Optional[str] = Header(None)) -> str:
    """Enforces Bearer token authentication for worker control plane."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header scheme. Expected 'Bearer <token>'",
        )
    token = parts[1]
    config = get_orchestrator_config()

    # 1. Check orchestrator config token
    if config.webhook_token and token == config.webhook_token:
        return token

    # 2. Check tenant profiles config
    try:
        profiles = list_camoufox_profiles(config.profiles_dir)
        for p in profiles:
            p_creds = get_profile_credentials(p["id"], config.profiles_dir)
            if p_creds.get("webhook_token") == token:
                return token
    except Exception as e:
        logger.debug(f"Profile token check error: {e}")

    logger.warning("Unauthorized worker handshake attempt detected with invalid token.")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or unauthorized token",
    )


@router.post("/start", status_code=status.HTTP_200_OK)
async def start_worker(
    profile_id: Optional[str] = Query(None, description="Optional tenant profile identifier"),
):
    """
    Asynchronously boots the remote Kaggle GPU compute worker.
    Non-blocking: Spawns 'kaggle kernels push' via asyncio subprocess.
    Transitions state: OFFLINE -> BOOTING / AWAITING_TUNNEL.
    """
    wm = get_worker_manager()
    result = await wm.start_worker(profile_id=profile_id)
    return result


@router.get("/status", response_model=WorkerStatusResponse)
async def get_worker_status(
    profile_id: Optional[str] = Query(None, description="Optional tenant profile identifier"),
):
    """
    Returns the current connection and lifecycle state of the remote Kaggle worker for a profile:
    State: OFFLINE | BOOTING | AWAITING_TUNNEL | READY | ERROR
    Includes Cloudflare tunnel URL (if ready), live uptime, and error details.
    """
    wm = get_worker_manager()
    return wm.get_status(profile_id=profile_id)


@router.post("/stop", status_code=status.HTTP_200_OK)
async def stop_worker(
    profile_id: Optional[str] = Query(None, description="Optional tenant profile identifier"),
):
    """
    Cancels/shuts down the Kaggle worker for a profile to conserve weekly GPU quota.
    Transitions state to OFFLINE.
    """
    wm = get_worker_manager()
    result = await wm.stop_worker(profile_id=profile_id)
    return result


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def worker_webhook(
    payload: WorkerHandshakePayload,
    token: str = Depends(verify_bearer_token),
):
    """
    Kaggle Worker Handshake Webhook.
    Called by kaggle_kernel.py after acquiring an authenticated Cloudflare reverse tunnel.
    Transitions state to READY.
    """
    logger.info(
        f"Webhook handshake received: worker_id='{payload.worker_id}', "
        f"tunnel='{payload.tunnel_url}', status='{payload.status}'"
    )
    wm = get_worker_manager()
    await wm.register_webhook(payload)

    # Sync with queue_manager if available in current process
    try:
        import swapedev_service.main as main_mod
        if hasattr(main_mod, "queue_manager") and main_mod.queue_manager:
            main_mod.queue_manager.register_worker(payload)
    except Exception as e:
        logger.debug(f"Queue manager sync skipped: {e}")

    config = get_orchestrator_config()
    return {
        "status": "accepted",
        "worker_id": payload.worker_id,
        "tunnel_url": payload.tunnel_url,
        "concurrency_limit": config.concurrency_limit,
    }


@router.post("/heartbeat", status_code=status.HTTP_200_OK)
async def worker_heartbeat(
    payload: WorkerHeartbeatPayload,
    token: str = Depends(verify_bearer_token),
):
    """Worker periodic keepalive heartbeat."""
    wm = get_worker_manager()
    recorded = await wm.record_heartbeat(payload.worker_id, profile_id=payload.profile_id)
    try:
        import swapedev_service.main as main_mod
        if hasattr(main_mod, "queue_manager") and main_mod.queue_manager:
            main_mod.queue_manager.record_heartbeat(payload.worker_id)
    except Exception:
        pass

    if not recorded and wm.state != WorkerState.READY:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker not registered or offline")
    return {"status": "ok", "worker_id": payload.worker_id, "profile_id": payload.profile_id}
