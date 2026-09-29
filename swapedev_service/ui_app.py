"""
SwapeDev Web UI & Autonomous Gatekeeper Service
===============================================
Runs strictly on 0.0.0.0:8776 as an extended service for the Desi Camoufox multi-tenant ecosystem.

Invariants enforced:
1. Global Gatekeeper: Rejects all requests to /setup unless valid Kaggle credentials exist in .env.
2. Tenant Gatekeeper: Root / lists Camoufox profiles; /dashboard strictly requires a verified physical profile folder.
3. Concurrency=1: Aligned with the single-flight orchestrator design to protect Kaggle T4 VRAM.
4. Pure HTML/JS + TailwindCSS (via CDN) + Jinja2Templates (Zero Node/React).
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, Query, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from swapedev_service.config import (
    get_orchestrator_config,
    has_valid_credentials,
    get_credential_status,
    get_default_profile_id,
    get_profile_swapedev_dir,
    get_profile_credentials_path,
    get_profile_credentials,
    has_valid_profile_credentials,
    save_profile_credentials,
    get_profile_credential_status,
    save_credentials,
    list_camoufox_profiles,
    verify_profile_id,
    load_app_env,
)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [swapedev.ui]: %(message)s",
)
logger = logging.getLogger("swapedev.ui")

# Paths & Templates
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "ui" / "templates"
STATIC_DIR = BASE_DIR / "ui" / "static"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def resolve_request_profile_id(request: Request) -> Optional[str]:
    """
    Resolves active profile ID from request:
    Priority 1: Query parameter ?profile_id=...
    Priority 2: Cookie 'swapedev_profile_id'
    Priority 3: Default ecosystem profile (e.g. 'maxx')
    """
    pid = request.query_params.get("profile_id")
    if pid and pid.strip():
        return pid.strip()

    cid = request.cookies.get("swapedev_profile_id")
    if cid and cid.strip():
        return cid.strip()

    config = get_orchestrator_config()
    return get_default_profile_id(config.profiles_dir)


# =========================================================================
# The Global Gatekeeper Middleware (Profile-Isolated)
# =========================================================================

class GlobalGatekeeperMiddleware(BaseHTTPMiddleware):
    """
    Autonomous Profile-Scoped Config Check:
    On every request, verifies that the active profile has valid Kaggle credentials
    in its own profile storage (profiles/<profile_id>/swapedev/config.json).
    If missing or invalid, routes user immediately to /setup?profile_id={clean_id}.
    """
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Permitted unconfigured bypass paths
        is_bypass = (
            path == "/setup" or
            path == "/setup/save" or
            path == "/health" or
            path.startswith("/static/") or
            path.startswith("/favicon") or
            path.startswith("/api/")
        )

        if not is_bypass:
            config = get_orchestrator_config()
            target_pid = resolve_request_profile_id(request)
            if target_pid:
                verified = verify_profile_id(target_pid, config.profiles_dir)
                if verified:
                    clean_id = verified.name[8:] if verified.name.startswith("profile_") else verified.name
                    if not has_valid_profile_credentials(clean_id, config.profiles_dir):
                        logger.info(f"Unconfigured profile [{clean_id}] accessing '{path}'. Intercepting and routing to /setup.")
                        response = RedirectResponse(
                            url=f"/setup?profile_id={clean_id}",
                            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                        )
                        response.set_cookie(
                            key="swapedev_profile_id",
                            value=clean_id,
                            httponly=True,
                            samesite="lax",
                        )
                        return response

        return await call_next(request)


# Initialize FastAPI App
app = FastAPI(
    title="SwapeDev Web UI & Tenant Gatekeeper",
    description="Extended web service for Desi Camoufox multi-tenant ecosystem running on port 8776",
    version="1.1.0",
)

# Register Gatekeeper Middleware
app.add_middleware(GlobalGatekeeperMiddleware)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount Worker Control Plane API Routes
from swapedev_service.worker_routes import router as worker_router
app.include_router(worker_router)


# =========================================================================
# Route: Health & Status
# =========================================================================

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint reporting gatekeeper status and profile counts."""
    config = get_orchestrator_config()
    target_pid = resolve_request_profile_id(request)
    configured = has_valid_profile_credentials(target_pid, config.profiles_dir) if target_pid else False
    profiles = list_camoufox_profiles(config.profiles_dir)
    return {
        "status": "healthy",
        "service": "SwapeDev Web UI",
        "port": 8776,
        "active_profile": target_pid,
        "credentials_configured": configured,
        "profiles_detected_count": len(profiles),
        "profiles_root": str(config.profiles_dir),
    }


# =========================================================================
# Route: Setup & Credentials (Profile-Scoped Gatekeeper Interface)
# =========================================================================

@app.get("/setup", response_class=HTMLResponse)
async def setup_page(
    request: Request,
    profile_id: Optional[str] = Query(None),
    error: Optional[str] = None,
):
    """
    Renders the setup guide and credentials form strictly scoped to the active profile.
    No local host ~/.kaggle/kaggle.json prefetching.
    """
    config = get_orchestrator_config()
    target_pid = profile_id or resolve_request_profile_id(request)

    if not target_pid:
        return HTMLResponse("No Camoufox profile found. Please launch or create a Camoufox profile first.", status_code=400)

    verified = verify_profile_id(target_pid, config.profiles_dir)
    clean_id = verified.name[8:] if verified and verified.name.startswith("profile_") else (verified.name if verified else target_pid)

    creds = get_profile_credentials(clean_id, config.profiles_dir) if verified else {}
    cred_status = get_profile_credential_status(clean_id, config.profiles_dir) if verified else {
        "profile_id": clean_id,
        "valid": False,
        "has_username": False,
        "username_masked": "",
        "has_key": False,
        "key_masked": "",
        "has_upstash_url": False,
        "upstash_url_masked": "",
        "has_upstash_token": False,
        "upstash_token_masked": "",
        "has_webhook_token": False,
        "webhook_token_masked": "",
        "config_path": f"{config.profiles_dir}/{clean_id}/config.json",
    }

    response = templates.TemplateResponse(
        request=request,
        name="setup.html",
        context={
            "profile_id": clean_id,
            "cred_status": cred_status,
            "current_username": creds.get("KAGGLE_USERNAME", ""),
            "current_key": creds.get("KAGGLE_KEY", ""),
            "current_upstash_url": creds.get("UPSTASH_REDIS_REST_URL", ""),
            "current_upstash_token": creds.get("UPSTASH_REDIS_REST_TOKEN", ""),
            "current_webhook_token": creds.get("webhook_token", ""),
            "active_profile": clean_id,
            "alert_message": error,
            "alert_type": "error" if error else None,
        }
    )
    response.set_cookie(
        key="swapedev_profile_id",
        value=clean_id,
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/setup")
async def setup_save(
    request: Request,
    username: str = Form(...),
    key: str = Form(...),
    upstash_url: str = Form(""),
    upstash_token: str = Form(""),
    webhook_token: Optional[str] = Form(None),
    profile_id: Optional[str] = Form(None),
):
    """
    Saves credentials to profiles/<profile_id>/config.json securely (0600),
    and redirects user to /dashboard?profile_id={clean_id}.
    """
    config = get_orchestrator_config()
    target_pid = profile_id or resolve_request_profile_id(request)

    if not target_pid:
        return RedirectResponse(
            url="/setup?error=Missing+profile+context",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    verified = verify_profile_id(target_pid, config.profiles_dir)
    if not verified:
        return RedirectResponse(
            url=f"/setup?error=Profile+'{target_pid}'+does+not+exist+on+disk",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    clean_id = verified.name[8:] if verified.name.startswith("profile_") else verified.name
    u = username.strip()
    k = key.strip()
    up_url = upstash_url.strip()
    up_tok = upstash_token.strip()
    t = (webhook_token or "").strip()

    if not u or not k or not up_url or not up_tok:
        return RedirectResponse(
            url=f"/setup?profile_id={clean_id}&error=All+fields+(Username,+API+Key,+Upstash+URL,+Upstash+Token)+are+required",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    try:
        saved_path = save_profile_credentials(
            profile_id=clean_id,
            username=u,
            key=k,
            upstash_redis_rest_url=up_url,
            upstash_redis_rest_token=up_tok,
            webhook_token=t,
            profiles_dir=config.profiles_dir,
        )
        logger.info(f"Gatekeeper credentials for profile [{clean_id}] stored at {saved_path}")
        response = RedirectResponse(
            url=f"/dashboard?profile_id={clean_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.set_cookie(
            key="swapedev_profile_id",
            value=clean_id,
            httponly=True,
            samesite="lax",
        )
        return response
    except Exception as e:
        logger.error(f"Failed to persist credentials for [{clean_id}]: {e}", exc_info=True)
        return RedirectResponse(
            url=f"/setup?profile_id={clean_id}&error=Failed+to+save+credentials:+{str(e)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )


# =========================================================================
# Route: Root / (Zero Profile Selector — Auto-Route to Current Context)
# =========================================================================

@app.get("/")
async def root_auto_router(
    request: Request,
    profile_id: Optional[str] = Query(None),
):
    """
    Root / endpoint:
    Zero profile switch screen. Wherever the user is logged in, that is the default and only profile.
    Auto-routes to /dashboard?profile_id=... if configured, or /setup?profile_id=... if unconfigured.
    """
    config = get_orchestrator_config()
    target_pid = profile_id or resolve_request_profile_id(request)

    if not target_pid:
        return HTMLResponse("No Camoufox profile found. Please launch or create a Camoufox profile first.", status_code=404)

    verified = verify_profile_id(target_pid, config.profiles_dir)
    if not verified:
        return HTMLResponse(f"Profile '{target_pid}' does not exist on disk.", status_code=404)

    clean_id = verified.name[8:] if verified.name.startswith("profile_") else verified.name

    if has_valid_profile_credentials(clean_id, config.profiles_dir):
        dest = f"/dashboard?profile_id={clean_id}"
    else:
        dest = f"/setup?profile_id={clean_id}"

    response = RedirectResponse(url=dest, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="swapedev_profile_id",
        value=clean_id,
        httponly=True,
        samesite="lax",
    )
    return response


# =========================================================================
# Route: Profile-Scoped Dashboard Skeleton
# =========================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def profile_dashboard(
    request: Request,
    profile_id: Optional[str] = Query(None),
):
    """
    Profile-Scoped Dashboard:
    - Strictly bound to active profile context. Zero profile switch option.
    - If unconfigured, redirects to /setup?profile_id=...
    - Displays active tenant context: "SwapeDev Console — Operating as: [maxx]".
    - Renders the 3 required placeholders:
      1. Disabled button: 'Warm GPU (Offline)'
      2. Section: 'Character Library (Identity Vault)'
      3. Section: 'Video Swap Pipeline'
    """
    config = get_orchestrator_config()
    target_pid = profile_id or resolve_request_profile_id(request)

    if not target_pid:
        return RedirectResponse(url="/setup", status_code=status.HTTP_303_SEE_OTHER)

    verified = verify_profile_id(target_pid, config.profiles_dir)
    if not verified:
        return RedirectResponse(
            url=f"/setup?error=Profile+'{target_pid}'+not+found",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    clean_id = verified.name[8:] if verified.name.startswith("profile_") else verified.name

    if not has_valid_profile_credentials(clean_id, config.profiles_dir):
        response = RedirectResponse(
            url=f"/setup?profile_id={clean_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.set_cookie(
            key="swapedev_profile_id",
            value=clean_id,
            httponly=True,
            samesite="lax",
        )
        return response

    # Check lock status
    lock_file = verified.parent / f"{verified.name}.lock"
    is_locked = lock_file.exists() or (verified / ".lock").exists()

    response = templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "profile_id": clean_id,
            "profile_folder_name": verified.name,
            "profile_path": str(verified),
            "is_locked": is_locked,
            "active_profile": clean_id,
        }
    )

    response.set_cookie(
        key="swapedev_profile_id",
        value=clean_id,
        httponly=True,
        samesite="lax",
    )
    return response


# =========================================================================
# Application Runner
# =========================================================================

def main():
    import uvicorn
    logger.info("Starting SwapeDev Web UI & Gatekeeper on http://0.0.0.0:8776 ...")
    uvicorn.run(
        "swapedev_service.ui_app:app",
        host="0.0.0.0",
        port=8776,
        reload=False,
    )


if __name__ == "__main__":
    main()
