"""
SwapeDev Strict Single-Flight FIFO Queue Manager
================================================
Guarantees concurrency=1 execution to protect Kaggle T4 VRAM.
Enforces multi-tenant profile isolation, ReActor cache eviction
between profile transitions, and strict asset routing.
Strict Lockdown: Standalone executions are strictly forbidden.
"""

import asyncio
import logging
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from swapedev_service.config import OrchestratorConfig, get_orchestrator_config
from swapedev_service.schemas import (
    SwapJobRequest,
    SwapJobResponse,
    WorkerHandshakePayload,
    WorkerStatusResponse,
)
from swapedev_service.worker_client import ComfyUIWorkerClient, WorkerClientError
from swapedev_service.workflow import build_face_swap_workflow

logger = logging.getLogger("swapedev.queue_manager")


class QueueSecurityError(Exception):
    """Raised when profile isolation or path boundary is violated."""
    pass


class ProfileNotFoundError(QueueSecurityError):
    """Raised when target profile does not exist on disk."""
    pass


class WorkerUnavailableError(Exception):
    """Raised when no active Kaggle worker is connected."""
    pass


@dataclass
class JobRecord:
    job_id: str
    request: SwapJobRequest
    created_at: float = field(default_factory=time.time)
    status: str = "queued"  # queued, processing, completed, failed
    output_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    completed_event: asyncio.Event = field(default_factory=asyncio.Event)


class SingleFlightQueueManager:
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or get_orchestrator_config()
        self._queue: asyncio.Queue[JobRecord] = asyncio.Queue()
        self._jobs: Dict[str, JobRecord] = {}
        self._active_worker: Optional[WorkerHandshakePayload] = None
        self._worker_client: Optional[ComfyUIWorkerClient] = None
        self._worker_last_seen: float = 0.0
        self._active_job: Optional[JobRecord] = None
        self._last_profile_id: Optional[str] = None
        self._worker_loop_task: Optional[asyncio.Task] = None
        self._running = False

    def sanitize_profile_id(self, profile_id: str) -> str:
        if not profile_id or not isinstance(profile_id, str):
            raise QueueSecurityError("Missing mandatory profile identifier.")
        clean = profile_id.strip()
        if not clean:
            raise QueueSecurityError("Profile identifier cannot be empty.")
        if not clean.startswith("profile_"):
            clean = f"profile_{clean}"
        # Block directory traversal attempts
        if "/" in clean or "\\" in clean or ".." in clean:
            raise QueueSecurityError(f"Illegal profile identifier: {profile_id}")
        return clean

    def get_profile_dir(self, profile_id: str) -> Path:
        clean_id = self.sanitize_profile_id(profile_id)
        profile_path = (self.config.profiles_dir / clean_id).resolve()
        # Verify scoped strictly inside profiles_dir
        if not str(profile_path).startswith(str(self.config.profiles_dir.resolve())):
            raise QueueSecurityError(f"Profile path escapes root profiles directory: {profile_path}")
        return profile_path

    def get_profile_staging_dir(self, profile_id: str) -> Path:
        """
        Namespaced staging directory per profile:
        Ensures orchestrator temporary processing paths are strictly isolated per tenant.
        """
        clean_id = self.sanitize_profile_id(profile_id)
        profile_staging = (self.config.staging_dir / clean_id).resolve()
        if not str(profile_staging).startswith(str(self.config.staging_dir.resolve())):
            raise QueueSecurityError(f"Staging path escapes base staging directory: {profile_staging}")
        profile_staging.mkdir(parents=True, exist_ok=True)
        return profile_staging

    def verify_profile_exists(self, profile_id: str) -> Path:
        """
        Hard Filesystem Gate:
        Verifies that ../avido-browser-platform/profiles/{profile_id} actually exists.
        """
        clean_id = self.sanitize_profile_id(profile_id)
        profile_dir = self.get_profile_dir(clean_id)
        if not profile_dir.exists() or not profile_dir.is_dir():
            raise ProfileNotFoundError(f"Unauthorized: Profile does not exist: {clean_id}")
        return profile_dir

    def register_worker(self, payload: WorkerHandshakePayload) -> None:
        """Register or update active Kaggle worker from authenticated webhook."""
        if payload.status == "offline":
            logger.info(f"Worker {payload.worker_id} reported offline shutdown.")
            self._active_worker = None
            self._worker_client = None
            return

        self._active_worker = payload
        self._worker_last_seen = time.time()
        self._worker_client = ComfyUIWorkerClient(
            tunnel_url=payload.tunnel_url,
            timeout_seconds=self.config.job_timeout_seconds,
        )
        logger.info(
            f"Kaggle Worker registered: id={payload.worker_id}, tunnel={payload.tunnel_url}, "
            f"gpu={payload.gpu_info}, dataset_ready={payload.dataset_ready}"
        )

    def record_heartbeat(self, worker_id: str) -> bool:
        if self._active_worker and self._active_worker.worker_id == worker_id:
            self._worker_last_seen = time.time()
            return True
        return False

    def get_worker_status(self) -> WorkerStatusResponse:
        from swapedev_service.worker_manager import get_worker_manager, WorkerState
        wm = get_worker_manager()
        status_resp = wm.get_status()
        status_resp.active_job_id = self._active_job.job_id if self._active_job else None
        status_resp.queued_jobs_count = self._queue.qsize()
        return status_resp

    async def submit_job(self, request: SwapJobRequest) -> JobRecord:
        """
        Enqueues a swap job into the strict FIFO queue.
        Enforces Hard Filesystem Gate: Rejects immediately if profile directory does not exist.
        """
        clean_pid = self.sanitize_profile_id(request.profile_id)
        request.profile_id = clean_pid

        # Hard Filesystem Gate: Reject before touching queue
        self.verify_profile_exists(clean_pid)

        job_id = f"swap_{uuid.uuid4().hex[:12]}"
        job = JobRecord(
            job_id=job_id,
            request=request,
            created_at=time.time(),
            status="queued",
        )
        self._jobs[job_id] = job
        await self._queue.put(job)
        logger.info(f"Job {job_id} for profile {clean_pid} enqueued (queue size: {self._queue.qsize()})")
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def to_response(self, job: JobRecord) -> SwapJobResponse:
        # Calculate queue position if still queued
        position = 0
        if job.status == "queued":
            # Count elements ahead in queue
            position = 1
            for j in self._jobs.values():
                if j.status == "queued" and j.created_at < job.created_at:
                    position += 1

        return SwapJobResponse(
            job_id=job.job_id,
            profile_id=job.request.profile_id,
            status=job.status,
            queue_position=position,
            output_path=job.output_path,
            duration_seconds=job.duration_seconds,
            error=job.error,
            created_at=job.created_at,
        )

    async def start(self) -> None:
        """Starts the background single-flight consumer loop."""
        if self._running:
            return
        self._running = True
        self._worker_loop_task = asyncio.create_task(self._process_queue_loop())
        logger.info("SwapeDev Single-Flight Queue Processor started.")

    async def stop(self) -> None:
        self._running = False
        if self._worker_loop_task:
            self._worker_loop_task.cancel()
            try:
                await self._worker_loop_task
            except asyncio.CancelledError:
                pass
        logger.info("SwapeDev Single-Flight Queue Processor stopped.")

    async def _process_queue_loop(self) -> None:
        """
        Strict single-flight FIFO execution loop (concurrency=1).
        Pulls jobs one by one, executes against remote Kaggle worker,
        evicts ReActor cache when profile changes, and writes to profile asset storage.
        """
        while self._running:
            try:
                job = await self._queue.get()
                self._active_job = job
                job.status = "processing"
                start_time = time.time()
                logger.info(f"Starting execution for job {job.job_id} (profile: {job.request.profile_id})")

                try:
                    await self._execute_job(job)
                    job.status = "completed"
                    job.duration_seconds = round(time.time() - start_time, 2)
                    logger.info(f"Job {job.job_id} completed successfully in {job.duration_seconds}s")
                except Exception as e:
                    job.status = "failed"
                    job.error = str(e)
                    job.duration_seconds = round(time.time() - start_time, 2)
                    logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
                finally:
                    self._active_job = None
                    job.completed_event.set()
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _execute_job(self, job: JobRecord) -> None:
        """Executes a single job against the connected worker."""
        if not self._worker_client or not self._active_worker:
            raise WorkerUnavailableError("No active Kaggle ComfyUI worker connected.")

        req = job.request
        profile_dir = self.verify_profile_exists(req.profile_id)
        identity_dir = profile_dir / "identity"
        assets_swaps_dir = profile_dir / "assets" / "swaps"
        assets_swaps_dir.mkdir(parents=True, exist_ok=True)

        # 1. Profile Isolation & ReActor Face Cache Clearing
        # If chaining jobs from different profiles, purge ComfyUI face cache and model tensors!
        if self._last_profile_id is not None and self._last_profile_id != req.profile_id:
            logger.info(
                f"Multi-tenant boundary switch detected ({self._last_profile_id} -> {req.profile_id}). "
                f"Flushing ComfyUI / ReActor face cache to prevent cross-contamination..."
            )
            await self._worker_client.flush_reactor_cache(unload_models=True)

        # 2. Resolve Input Media Path
        if not req.input_path:
            raise ValueError("Job request must specify input_path.")

        input_path = Path(req.input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input media path does not exist: {input_path}")

        # 3. Resolve Target Face (Identity Vault)
        target_face_path: Optional[Path] = None
        if req.target_face_path:
            target_face_path = Path(req.target_face_path).resolve()
        else:
            # Auto-lookup in profile identity vault
            possible_faces = [
                identity_dir / "source_face.png",
                identity_dir / "source_face.jpg",
                identity_dir / "identity_face.png",
                identity_dir / "identity_face.jpg",
            ]
            for p in possible_faces:
                if p.exists():
                    target_face_path = p
                    break

        if not target_face_path or not target_face_path.exists():
            raise FileNotFoundError(
                f"No source face found for profile {req.profile_id}. "
                f"Expected target_face_path or image in {identity_dir}"
            )

        # 4. Upload Assets to Worker ComfyUI
        # Assign run-unique remote names to prevent ComfyUI file cache collision
        remote_input_name = f"{job.job_id}_input_{input_path.name}"
        remote_face_name = f"{job.job_id}_face_{target_face_path.name}"

        logger.info(f"Uploading input media {input_path.name} to worker...")
        await self._worker_client.upload_file(input_path, remote_input_name)

        logger.info(f"Uploading identity face {target_face_path.name} to worker...")
        await self._worker_client.upload_file(target_face_path, remote_face_name)

        # 5. Build ComfyUI Workflow
        workflow = build_face_swap_workflow(
            input_image_name=remote_input_name,
            source_face_image_name=remote_face_name,
            params=req.parameters,
            output_prefix=f"{req.profile_id}_{job.job_id}",
        )

        # 6. Queue Prompt in ComfyUI
        prompt_id = await self._worker_client.queue_prompt(workflow, client_id=f"client_{req.profile_id}")
        logger.info(f"Job {job.job_id} submitted to ComfyUI as prompt {prompt_id}")

        # 7. Poll Execution History
        outputs = await self._worker_client.poll_execution(prompt_id)
        logger.info(f"ComfyUI prompt {prompt_id} execution finished. Harvesting outputs...")

        # 8. Locate Rendered Output Node Image
        rendered_images = []
        for node_id, node_out in outputs.items():
            if "images" in node_out:
                for img_info in node_out["images"]:
                    rendered_images.append(img_info)

        if not rendered_images:
            raise WorkerClientError(f"No rendered images found in ComfyUI outputs: {outputs}")

        primary_image = rendered_images[0]
        out_filename = primary_image.get("filename")
        subfolder = primary_image.get("subfolder", "")
        type_name = primary_image.get("type", "output")

        # 9. Strict Routing to Scoped Profile Assets Directory
        final_dest_filename = f"{job.job_id}_{out_filename}"
        final_dest_path = (assets_swaps_dir / final_dest_filename).resolve()

        # Security check: verify final destination is strictly within profile directory
        if not str(final_dest_path).startswith(str(profile_dir)):
            raise QueueSecurityError(f"Output path escapes tenant profile scope: {final_dest_path}")

        logger.info(f"Downloading output asset to tenant storage: {final_dest_path}")
        await self._worker_client.download_output(
            filename=out_filename,
            subfolder=subfolder,
            type_name=type_name,
            target_path=final_dest_path,
        )

        job.output_path = str(final_dest_path)
        self._last_profile_id = req.profile_id

        # 10. Post-job memory cleanup
        await self._worker_client.flush_reactor_cache(unload_models=False)
