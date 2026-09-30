"""
SwapeDev Video Worker Client
============================
Async HTTP client for interacting with the remote Kaggle Video Character Inpainting
FastAPI worker over reverse tunnel (Cloudflare Quick Tunnel / pyngrok).
"""

import asyncio
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Union
import httpx

logger = logging.getLogger("swapedev.video_worker_client")


class VideoWorkerClientError(Exception):
    """Exception raised when communication with the video inpainting worker fails."""
    pass


class VideoInpaintingWorkerClient:
    """
    Client for dispatching character replacement jobs to the remote Kaggle video worker.
    """

    def __init__(self, tunnel_url: str, timeout_seconds: float = 600.0):
        self.tunnel_url = tunnel_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def is_healthy(self) -> bool:
        """Checks if the remote video inpainting service is alive."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.tunnel_url}/health")
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for video worker {self.tunnel_url}: {e}")
            return False

    async def get_gpu_status(self) -> Dict[str, Any]:
        """Queries GPU status and VRAM utilization."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.tunnel_url}/gpu-status")
                if res.status_code == 200:
                    return res.json()
        except Exception as e:
            logger.warning(f"Failed to query GPU status: {e}")
        return {"cuda_available": False, "error": "Unreachable"}

    async def clear_vram(self) -> bool:
        """Manually requests remote GPU VRAM eviction."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(f"{self.tunnel_url}/vram/clear")
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to clear remote VRAM: {e}")
            return False

    async def process_shot(
        self,
        video_url: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        prompt: str = "spiderman suit, highly detailed, cinematic lighting, marvel superhero, 8k",
        negative_prompt: str = "deformed, blurry, bad anatomy, human face, glitch, boiling artifacts",
        character_lora: str = "spiderman",
        denoise: float = 0.80,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        lora_weight: float = 0.85,
        seed: Optional[int] = None,
        return_format: str = "url",
    ) -> Dict[str, Any]:
        """
        Dispatches a shot inpainting job to POST /process-shot on the Kaggle worker.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "character_lora": character_lora,
            "denoise": denoise,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "lora_weight": lora_weight,
            "seed": seed,
            "return_format": return_format,
        }

        if video_url:
            payload["video_url"] = video_url
        elif video_path:
            p = Path(video_path)
            if not p.exists():
                raise VideoWorkerClientError(f"Local input video does not exist: {p}")
            with open(p, "rb") as f:
                payload["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
        else:
            raise VideoWorkerClientError("Either video_url or video_path must be specified.")

        logger.info(f"Submitting video shot to {self.tunnel_url}/process-shot...")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                res = await client.post(f"{self.tunnel_url}/process-shot", json=payload)
                if res.status_code != 200:
                    raise VideoWorkerClientError(
                        f"Worker returned HTTP {res.status_code}: {res.text}"
                    )
                return res.json()
        except httpx.TimeoutException:
            raise VideoWorkerClientError(
                f"Video inpainting request timed out after {self.timeout_seconds}s"
            )
        except Exception as e:
            raise VideoWorkerClientError(f"Request error: {e}") from e

    async def download_output_video(self, video_url: str, target_local_path: Union[str, Path]) -> Path:
        """Downloads rendered output video MP4 from worker to local disk."""
        target_path = Path(target_local_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("GET", video_url) as response:
                if response.status_code != 200:
                    raise VideoWorkerClientError(
                        f"Failed to download video from {video_url}: HTTP {response.status_code}"
                    )
                with open(target_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        logger.info(f"Downloaded output video to: {target_path}")
        return target_path
