"""
SwapeDev Worker Client
======================
Handles communication with the remote Kaggle ComfyUI server over
the authenticated reverse tunnel. Implements upload, prompt execution,
output download, and explicit VRAM/cache flushing.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger("swapedev.worker_client")


class WorkerClientError(Exception):
    """Base exception for remote worker communication failures."""
    pass


class ComfyUIWorkerClient:
    def __init__(self, tunnel_url: str, timeout_seconds: int = 300):
        self.tunnel_url = tunnel_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def is_healthy(self) -> bool:
        """Check if remote ComfyUI instance is alive and reachable."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.tunnel_url}/system_stats")
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for {self.tunnel_url}: {e}")
            return False

    async def upload_file(self, file_path: Path, remote_filename: Optional[str] = None) -> str:
        """
        Uploads an image or media asset to ComfyUI's input directory.
        Returns the resolved remote filename.
        """
        if not file_path.exists():
            raise WorkerClientError(f"File to upload does not exist: {file_path}")

        upload_name = remote_filename or file_path.name

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(file_path, "rb") as f:
                    files = {
                        "image": (upload_name, f, "application/octet-stream")
                    }
                    data = {
                        "overwrite": "true",
                        "type": "input",
                    }
                    res = await client.post(
                        f"{self.tunnel_url}/upload/image",
                        files=files,
                        data=data,
                    )
                if res.status_code != 200:
                    raise WorkerClientError(f"Failed to upload image: {res.status_code} {res.text}")

                res_data = res.json()
                return res_data.get("name", upload_name)
        except Exception as e:
            raise WorkerClientError(f"Upload error: {e}") from e

    async def queue_prompt(self, workflow: Dict[str, Any], client_id: str = "swapedev_orchestrator") -> str:
        """
        Dispatches workflow graph to ComfyUI /prompt endpoint.
        Returns the assigned prompt_id.
        """
        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(f"{self.tunnel_url}/prompt", json=payload)
                if res.status_code != 200:
                    raise WorkerClientError(f"Failed to queue prompt: {res.status_code} {res.text}")

                data = res.json()
                if "error" in data:
                    raise WorkerClientError(f"ComfyUI prompt error: {data['error']}")
                prompt_id = data.get("prompt_id")
                if not prompt_id:
                    raise WorkerClientError(f"Missing prompt_id in ComfyUI response: {data}")
                return prompt_id
        except Exception as e:
            raise WorkerClientError(f"Queue prompt error: {e}") from e

    async def poll_execution(self, prompt_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        """
        Polls ComfyUI /history/{prompt_id} until the job completes or fails.
        Returns the output node outputs.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.timeout_seconds:
                raise WorkerClientError(f"Job execution timed out after {self.timeout_seconds}s for prompt {prompt_id}")

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    res = await client.get(f"{self.tunnel_url}/history/{prompt_id}")
                    if res.status_code == 200:
                        data = res.json()
                        if prompt_id in data:
                            history = data[prompt_id]
                            # Check status
                            status_info = history.get("status", {})
                            if status_info.get("status_str") == "error":
                                messages = status_info.get("messages", [])
                                raise WorkerClientError(f"ComfyUI job failed: {messages}")

                            outputs = history.get("outputs", {})
                            if outputs:
                                return outputs
            except WorkerClientError:
                raise
            except Exception as e:
                logger.debug(f"Polling history warning for {prompt_id}: {e}")

            await asyncio.sleep(poll_interval)

    async def download_output(
        self,
        filename: str,
        subfolder: str,
        type_name: str,
        target_path: Path,
    ) -> Path:
        """
        Downloads rendered output asset from ComfyUI /view endpoint directly to target_path.
        """
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": type_name,
        }

        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("GET", f"{self.tunnel_url}/view", params=params) as res:
                    if res.status_code != 200:
                        raise WorkerClientError(f"Failed to download asset {filename}: {res.status_code}")

                    with open(target_path, "wb") as f:
                        async for chunk in res.aiter_bytes():
                            f.write(chunk)

            return target_path
        except Exception as e:
            raise WorkerClientError(f"Download output error: {e}") from e

    async def flush_reactor_cache(self, unload_models: bool = True) -> bool:
        """
        Enforce strict isolation: calls ComfyUI's /free endpoint to purge
        all cached face embeddings, InsightFace analysis caches, and GPU VRAM.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                payload = {
                    "unload_models": unload_models,
                    "free_memory": True,
                }
                res = await client.post(f"{self.tunnel_url}/free", json=payload)
                if res.status_code == 200:
                    logger.info("Successfully flushed ComfyUI / ReActor cache & freed VRAM.")
                    return True
                else:
                    logger.warning(f"ComfyUI /free returned {res.status_code}: {res.text}")
                    return False
        except Exception as e:
            logger.warning(f"Failed to flush ComfyUI cache: {e}")
            return False
