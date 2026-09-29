"""
SwapeDev Global Orchestrator Service
====================================
FastAPI REST API managing:
- Secure Kaggle Worker Webhook Handshake (Bearer Auth)
- Strict Single-Flight FIFO Queue (concurrency=1)
- Mandatory Profile Validation & Hard Filesystem Gate (HTTP 403 if profile missing)
- Zero Standalone / Orphan Execution: Everything strictly bound to Desi Camoufox profiles.
"""

import os
import asyncio
import logging
import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from swapedev_service.config import get_orchestrator_config, OrchestratorConfig
from swapedev_service.schemas import (
    WorkerHandshakePayload,
    WorkerHeartbeatPayload,
    WorkerStatusResponse,
    SwapJobRequest,
    SwapJobResponse,
    SwapJobParams,
)
from swapedev_service.queue_manager import (
    SingleFlightQueueManager,
    QueueSecurityError,
    ProfileNotFoundError,
    WorkerUnavailableError,
)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("swapedev.orchestrator")

config = get_orchestrator_config()
queue_manager = SingleFlightQueueManager(config=config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start single-flight queue worker loop
    await queue_manager.start()
    yield
    # Shutdown: Stop worker loop cleanly
    await queue_manager.stop()


app = FastAPI(
    title="SwapeDev Remote Compute Orchestrator",
    description="Multi-tenant single-flight GPU compute orchestrator for Desi Camoufox & Kaggle ComfyUI (Locked Down)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    if token != config.webhook_token:
        logger.warning("Unauthorized worker handshake attempt detected with invalid token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized token",
        )
    return token


# =========================================================================
# Worker Control Plane Endpoints
# =========================================================================
from swapedev_service.worker_routes import router as worker_router
app.include_router(worker_router)


# =========================================================================
# Strict Tenant-Only Job Submission Endpoints (Hard Filesystem Gated)
# =========================================================================

@app.post("/api/jobs/swap", response_model=SwapJobResponse)
async def submit_swap_job(
    request: SwapJobRequest,
):
    """
    Submits a face swap job to the strict single-flight FIFO queue.
    Hard Filesystem Gate: Rejects immediately with HTTP 403 Forbidden if profile_id does not exist on disk.
    """
    # 1. Hard Filesystem Gate Validation
    try:
        clean_pid = queue_manager.sanitize_profile_id(request.profile_id)
        queue_manager.verify_profile_exists(clean_pid)
    except (ProfileNotFoundError, QueueSecurityError) as e:
        logger.warning(f"Admission control rejection: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized: Profile does not exist.",
        )

    # 2. Enqueue in single-flight FIFO queue
    try:
        job = await queue_manager.submit_job(request)
    except ProfileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized: Profile does not exist.")
    except QueueSecurityError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to enqueue job: {e}")

    # 3. Synchronous wait if requested
    if request.wait:
        try:
            await asyncio.wait_for(job.completed_event.wait(), timeout=float(config.job_timeout_seconds))
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Job {job.job_id} timed out after {config.job_timeout_seconds} seconds",
            )

    return queue_manager.to_response(job)


@app.post("/api/jobs/swap/upload", response_model=SwapJobResponse)
async def submit_swap_job_multipart(
    profile_id: str = Form(..., description="Mandatory tenant profile identifier"),
    input_file: UploadFile = File(...),
    target_face_file: Optional[UploadFile] = File(None),
    denoise: float = Form(0.6),
    face_restore_model: str = Form("GFPGANv1.4"),
    codeformer_weight: float = Form(0.5),
    wait: bool = Form(True),
):
    """
    Multipart upload endpoint strictly requiring a valid profile_id.
    Hard Filesystem Gate: Rejects immediately with HTTP 403 if profile does not exist.
    Staging is strictly namespaced inside the tenant profile to prevent cross-talk.
    """
    # Hard Filesystem Gate
    try:
        clean_pid = queue_manager.sanitize_profile_id(profile_id)
        profile_dir = queue_manager.verify_profile_exists(clean_pid)
    except (ProfileNotFoundError, QueueSecurityError) as e:
        logger.warning(f"Multipart admission control rejection: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized: Profile does not exist.",
        )

    # Isolated staging strictly within tenant profile directory
    staging_dir = queue_manager.get_profile_staging_dir(clean_pid)

    unique_prefix = uuid.uuid4().hex[:8]
    input_staging_path = staging_dir / f"{unique_prefix}_{input_file.filename}"
    with open(input_staging_path, "wb") as f:
        shutil.copyfileobj(input_file.file, f)

    target_face_path = None
    if target_face_file:
        target_staging_path = staging_dir / f"{unique_prefix}_{target_face_file.filename}"
        with open(target_staging_path, "wb") as f:
            shutil.copyfileobj(target_face_file.file, f)
        target_face_path = str(target_staging_path)

    params = SwapJobParams(
        denoise=denoise,
        face_restore_model=face_restore_model,
        codeformer_weight=codeformer_weight,
    )

    req = SwapJobRequest(
        profile_id=clean_pid,
        input_path=str(input_staging_path),
        target_face_path=target_face_path,
        parameters=params,
        wait=wait,
    )

    return await submit_swap_job(req)


@app.get("/api/jobs/{job_id}", response_model=SwapJobResponse)
async def get_job_status(job_id: str):
    """Retrieves status, progress, queue position, or output path of a specific job."""
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return queue_manager.to_response(job)


@app.get("/health")
async def health_check():
    """System health check endpoint."""
    status_info = queue_manager.get_worker_status()
    return {
        "status": "healthy",
        "service": "SwapeDev Orchestrator",
        "concurrency_limit": config.concurrency_limit,
        "worker_connected": status_info.connected,
        "queue_size": status_info.queued_jobs_count,
    }


def main():
    import uvicorn
    uvicorn.run(
        "swapedev_service.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
