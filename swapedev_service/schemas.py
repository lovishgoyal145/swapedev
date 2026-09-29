"""
SwapeDev Orchestrator Data Contracts & Schemas
==============================================
Pydantic models for worker registration, jobs, and status queries.
Strict multi-tenant lockdown: No standalone or unauthenticated execution permitted.
"""

import time
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class WorkerHandshakePayload(BaseModel):
    worker_id: str = Field(..., min_length=1)
    tunnel_url: str = Field(..., min_length=1)
    profile_id: Optional[str] = None
    status: str = "online"
    gpu_info: Optional[str] = None
    dataset_ready: bool = True
    timestamp: float = Field(default_factory=time.time)


class WorkerHeartbeatPayload(BaseModel):
    worker_id: str = Field(..., min_length=1)
    profile_id: Optional[str] = None
    status: str = "online"
    active_jobs: int = 0
    uptime_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class SwapJobParams(BaseModel):
    denoise: float = 0.6
    face_restore_model: str = "GFPGANv1.4"
    codeformer_weight: float = 0.5
    detect_gender_source: str = "no"
    detect_gender_input: str = "no"
    source_faces_index: int = 0
    input_faces_index: int = 0
    seed: Optional[int] = None
    prompt: Optional[str] = None
    extra_options: Dict[str, Any] = Field(default_factory=dict)


class SwapJobRequest(BaseModel):
    # Mandatory Profile Validation: profile_id is strict and required (no default values)
    profile_id: str = Field(
        ...,
        min_length=1,
        description="Mandatory tenant profile ID. Standalone or unassociated requests are strictly forbidden."
    )
    input_path: str = Field(
        ...,
        min_length=1,
        description="Mandatory input media path scoped to the tenant profile."
    )
    target_face_path: Optional[str] = None
    parameters: SwapJobParams = Field(default_factory=SwapJobParams)
    wait: bool = True


class SwapJobResponse(BaseModel):
    job_id: str
    profile_id: str
    status: str  # "queued", "processing", "completed", "failed"
    queue_position: int = 0
    output_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


class WorkerStatusResponse(BaseModel):
    state: str = "OFFLINE"
    profile_id: Optional[str] = None
    connected: bool = False
    worker_id: Optional[str] = None
    tunnel_url: Optional[str] = None
    status: str = "disconnected"
    uptime: Optional[float] = 0.0
    error: Optional[str] = None
    last_heartbeat: Optional[float] = None
    active_job_id: Optional[str] = None
    queued_jobs_count: int = 0
