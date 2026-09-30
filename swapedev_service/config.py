"""
SwapeDev Orchestrator Configuration
===================================
Manages system-level settings, authentication tokens, profile storage roots,
concurrency constraints, autonomous gatekeeper credential verification,
and Desi Camoufox tenant discovery.
"""

import os
import json
import secrets
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger("swapedev.config")


def get_canonical_env_path() -> Path:
    """Returns the primary canonical path for the .env configuration file."""
    explicit = os.getenv("SWAPEDEV_ENV_FILE")
    if explicit:
        return Path(explicit).resolve()

    # Priority 1: Service directory .env
    service_env = Path(__file__).resolve().parent / ".env"
    if service_env.exists():
        return service_env

    # Priority 2: SwapeDev root directory .env
    swapedev_env = Path(__file__).resolve().parents[1] / "SwapeDev" / ".env"
    if swapedev_env.exists():
        return swapedev_env

    # Priority 3: Current working directory .env
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env

    # Default to service directory
    return service_env


def load_app_env() -> Dict[str, str]:
    """Loads environment variables from canonical .env into os.environ."""
    env_path = get_canonical_env_path()
    if env_path.exists():
        load_dotenv(env_path, override=True)
        return {k: v for k, v in dotenv_values(env_path).items() if v is not None}
    return {}


# Load environment on module import
load_app_env()


@dataclass
class OrchestratorConfig:
    host: str = field(
        default_factory=lambda: os.getenv("SWAPEDEV_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("SWAPEDEV_PORT", "8000"))
    )
    ui_port: int = field(
        default_factory=lambda: int(os.getenv("SWAPEDEV_UI_PORT", "8776"))
    )
    # Auth token for Kaggle worker handshake webhook
    webhook_token: str = field(
        default_factory=lambda: os.getenv("SWAPEDEV_WEBHOOK_TOKEN", "")
    )
    # Kaggle API credentials
    kaggle_username: str = field(
        default_factory=lambda: os.getenv("KAGGLE_USERNAME", "")
    )
    kaggle_key: str = field(
        default_factory=lambda: os.getenv("KAGGLE_KEY", "")
    )
    # Base profiles directory (Desi Camoufox ecosystem)
    profiles_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "AVIDO_PROFILES_DIR",
                Path(__file__).resolve().parents[1] / "avido-browser-platform" / "profiles",
            )
        ).resolve()
    )
    # Strict single-flight FIFO concurrency to protect Kaggle T4 VRAM
    concurrency_limit: int = 1
    # Timeouts
    job_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("SWAPEDEV_JOB_TIMEOUT_SECONDS", "720"))
    )
    idle_watchdog_seconds: int = field(
        default_factory=lambda: int(os.getenv("SWAPEDEV_IDLE_WATCHDOG_SECONDS", "600"))
    )
    # Temporary staging directory for orchestrator
    staging_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SWAPEDEV_STAGING_DIR",
                Path(__file__).resolve().parent / "staging",
            )
        ).resolve()
    )

    def validate(self) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if self.concurrency_limit != 1:
            raise ValueError("SwapeDev requires strict concurrency_limit = 1 to protect Kaggle T4 VRAM.")


_global_config: Optional[OrchestratorConfig] = None


def get_orchestrator_config(reload: bool = False) -> OrchestratorConfig:
    global _global_config
    if _global_config is None or reload:
        load_app_env()
        _global_config = OrchestratorConfig()
        _global_config.validate()
    return _global_config


def set_orchestrator_config(config: OrchestratorConfig) -> None:
    global _global_config
    config.validate()
    _global_config = config


def reset_orchestrator_config() -> None:
    """Resets the cached global orchestrator configuration so it will be reloaded on next access."""
    global _global_config
    _global_config = None


# =========================================================================
# Profile-Isolated Credential & Autonomous Gatekeeper Helpers
# =========================================================================

def get_default_profile_id(profiles_dir: Optional[Path] = None) -> Optional[str]:
    """
    Returns the default active profile ID for the ecosystem.
    Prioritizes 'maxx' if present, otherwise the first discovered profile.
    """
    profiles = list_camoufox_profiles(profiles_dir)
    if not profiles:
        return None
    for p in profiles:
        if p["id"].lower() == "maxx":
            return p["id"]
    return profiles[0]["id"]


def get_profile_swapedev_dir(profile_id: str, profiles_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Returns the SwapeDev service storage directory inside the specified Camoufox profile.
    Ensures directory exists.
    """
    verified = verify_profile_id(profile_id, profiles_dir)
    if not verified:
        return None
    swapedev_dir = verified / "swapedev"
    swapedev_dir.mkdir(parents=True, exist_ok=True)
    return swapedev_dir


def get_profile_credentials_path(profile_id: str, profiles_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Returns path to config.json for the specified profile.
    Priority 1: profiles/{profile_id}/config.json
    Priority 2: profiles/{profile_id}/swapedev/config.json (legacy fallback)
    """
    verified = verify_profile_id(profile_id, profiles_dir)
    if not verified:
        return None
    root_cfg = verified / "config.json"
    if root_cfg.exists() and root_cfg.is_file():
        return root_cfg
    legacy_cfg = verified / "swapedev" / "config.json"
    if legacy_cfg.exists() and legacy_cfg.is_file():
        return legacy_cfg
    return root_cfg


def get_profile_credentials(profile_id: str, profiles_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Loads credentials strictly from the profile's config.json.
    Zero cross-profile or global host leakage.
    Supports KAGGLE_USERNAME, KAGGLE_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN.
    """
    cfg_path = get_profile_credentials_path(profile_id, profiles_dir)
    if cfg_path and cfg_path.exists() and cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            k_user = str(data.get("KAGGLE_USERNAME") or data.get("kaggle_username") or "").strip()
            k_key = str(data.get("KAGGLE_KEY") or data.get("kaggle_key") or "").strip()
            up_url = str(data.get("UPSTASH_REDIS_REST_URL") or data.get("upstash_redis_rest_url") or data.get("UPSTASH_REST_URL") or data.get("upstash_rest_url") or "").strip()
            up_token = str(data.get("UPSTASH_REDIS_REST_TOKEN") or data.get("upstash_redis_rest_token") or data.get("UPSTASH_REST_TOKEN") or data.get("upstash_rest_token") or "").strip()
            webhook_token = str(data.get("webhook_token") or "").strip()

            if not k_user or not k_key:
                env_u = os.getenv("KAGGLE_USERNAME", "").strip()
                env_k = os.getenv("KAGGLE_KEY", "").strip()
                if env_u and env_k:
                    logger.warning(
                        f"LOUD WARNING: Profile '{profile_id}' config.json is missing Kaggle credentials! "
                        f"Found credentials in host environment (.env). Global fallback is strictly disallowed to preserve multi-tenant isolation!"
                    )

            return {
                "kaggle_username": k_user,
                "kaggle_key": k_key,
                "upstash_redis_rest_url": up_url,
                "upstash_redis_rest_token": up_token,
                "KAGGLE_USERNAME": k_user,
                "KAGGLE_KEY": k_key,
                "UPSTASH_REDIS_REST_URL": up_url,
                "UPSTASH_REDIS_REST_TOKEN": up_token,
                "webhook_token": webhook_token,
            }
        except Exception as e:
            logger.warning(f"Failed to read credentials for profile '{profile_id}' from {cfg_path}: {e}")
    else:
        logger.warning(f"Profile credentials file not found for profile '{profile_id}' at {cfg_path}")
        env_u = os.getenv("KAGGLE_USERNAME", "").strip()
        env_k = os.getenv("KAGGLE_KEY", "").strip()
        if env_u and env_k:
            logger.warning(
                f"LOUD WARNING: Profile '{profile_id}' has no config.json! Host environment has KAGGLE credentials, "
                f"but SwapeDev strictly enforces tenant-isolated credentials. Zero fallback to .env allowed!"
            )
    return {
        "kaggle_username": "",
        "kaggle_key": "",
        "upstash_redis_rest_url": "",
        "upstash_redis_rest_token": "",
        "KAGGLE_USERNAME": "",
        "KAGGLE_KEY": "",
        "UPSTASH_REDIS_REST_URL": "",
        "UPSTASH_REDIS_REST_TOKEN": "",
        "webhook_token": "",
    }


def has_valid_profile_credentials(profile_id: str, profiles_dir: Optional[Path] = None) -> bool:
    """
    Verifies that the given profile has non-empty, non-placeholder credentials
    in its own config.json:
    - KAGGLE_USERNAME
    - KAGGLE_KEY
    - UPSTASH_REDIS_REST_URL
    - UPSTASH_REDIS_REST_TOKEN
    """
    creds = get_profile_credentials(profile_id, profiles_dir)
    username = creds.get("KAGGLE_USERNAME", "").strip()
    key = creds.get("KAGGLE_KEY", "").strip()
    upstash_url = creds.get("UPSTASH_REDIS_REST_URL", "").strip()
    upstash_token = creds.get("UPSTASH_REDIS_REST_TOKEN", "").strip()

    if not username or not key or not upstash_url or not upstash_token:
        return False

    placeholders = {
        "your_username", "your_kaggle_key", "your_token", "placeholder", "xxx",
        "your_upstash_rest_token", "https://your-upstash-db.upstash.io",
    }
    if (
        username.lower() in placeholders
        or key.lower() in placeholders
        or upstash_token.lower() in placeholders
        or "your-upstash" in upstash_url.lower()
    ):
        return False

    return True


def save_profile_credentials(
    profile_id: str,
    username: str,
    key: str,
    upstash_redis_rest_url: str = "",
    upstash_redis_rest_token: str = "",
    webhook_token: Optional[str] = None,
    profiles_dir: Optional[Path] = None,
) -> Path:
    """
    Persists credentials strictly to profiles/<profile_id>/config.json with chmod 0600.
    Also mirrors to profiles/<profile_id>/swapedev/config.json for sub-system compatibility.
    Initializes tenant subdirectories: character_vault/ and staging/.
    """
    verified = verify_profile_id(profile_id, profiles_dir)
    if not verified:
        raise ValueError(f"Cannot save credentials: Profile '{profile_id}' does not exist on disk.")

    s_dir = verified / "swapedev"
    s_dir.mkdir(parents=True, exist_ok=True)
    (s_dir / "character_vault").mkdir(parents=True, exist_ok=True)
    (s_dir / "staging").mkdir(parents=True, exist_ok=True)

    clean_username = username.strip()
    clean_key = key.strip()
    clean_upstash_url = upstash_redis_rest_url.strip()
    clean_upstash_token = upstash_redis_rest_token.strip()
    clean_token = (webhook_token or "").strip()
    if not clean_token:
        clean_token = f"swp_sec_{secrets.token_urlsafe(24)}"

    clean_id = verified.name[8:] if verified.name.startswith("profile_") else verified.name
    cfg_data = {
        "profile_id": clean_id,
        "KAGGLE_USERNAME": clean_username,
        "KAGGLE_KEY": clean_key,
        "UPSTASH_REDIS_REST_URL": clean_upstash_url,
        "UPSTASH_REDIS_REST_TOKEN": clean_upstash_token,
        # Backward-compatible lowercase aliases
        "kaggle_username": clean_username,
        "kaggle_key": clean_key,
        "upstash_redis_rest_url": clean_upstash_url,
        "upstash_redis_rest_token": clean_upstash_token,
        "webhook_token": clean_token,
        "updated_at": datetime.now().isoformat(),
    }

    # Primary config: profiles/{profile_id}/config.json
    root_cfg_path = verified / "config.json"
    with open(root_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)
    try:
        os.chmod(root_cfg_path, 0o600)
    except Exception as e:
        logger.warning(f"Could not chmod 600 on {root_cfg_path}: {e}")

    # Mirror to profiles/{profile_id}/swapedev/config.json
    legacy_cfg_path = s_dir / "config.json"
    try:
        with open(legacy_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
        os.chmod(legacy_cfg_path, 0o600)
    except Exception as e:
        logger.warning(f"Could not write legacy mirror config: {e}")

    logger.info(f"Profile credentials saved to {root_cfg_path} for [{clean_id}]")
    return root_cfg_path


def get_profile_credential_status(profile_id: str, profiles_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Returns present status and masked preview of credentials for a given profile."""
    creds = get_profile_credentials(profile_id, profiles_dir)
    username = creds.get("KAGGLE_USERNAME", "")
    key = creds.get("KAGGLE_KEY", "")
    upstash_url = creds.get("UPSTASH_REDIS_REST_URL", "")
    upstash_token = creds.get("UPSTASH_REDIS_REST_TOKEN", "")
    token = creds.get("webhook_token", "")
    cfg_path = get_profile_credentials_path(profile_id, profiles_dir)

    def mask(val: str) -> str:
        if not val:
            return ""
        if len(val) <= 6:
            return "******"
        return f"{val[:3]}...{val[-3:]}"

    return {
        "profile_id": profile_id,
        "valid": has_valid_profile_credentials(profile_id, profiles_dir),
        "has_username": bool(username),
        "username_masked": mask(username),
        "has_key": bool(key),
        "key_masked": mask(key),
        "has_upstash_url": bool(upstash_url),
        "upstash_url_masked": mask(upstash_url),
        "has_upstash_token": bool(upstash_token),
        "upstash_token_masked": mask(upstash_token),
        "has_webhook_token": bool(token),
        "webhook_token_masked": mask(token),
        "config_path": str(cfg_path) if cfg_path else "Unknown",
    }


def has_valid_credentials(profile_id: Optional[str] = None) -> bool:
    """
    Verifies valid credentials for the given profile, or the default active profile.
    """
    target_pid = profile_id or get_default_profile_id()
    if target_pid:
        return has_valid_profile_credentials(target_pid)
    return False


def get_credential_status(profile_id: Optional[str] = None) -> Dict[str, Any]:
    """Returns credential status for given or default profile."""
    target_pid = profile_id or get_default_profile_id()
    if target_pid:
        return get_profile_credential_status(target_pid)
    return {
        "profile_id": None,
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
        "config_path": "No profile found",
    }


def save_credentials(
    username: str,
    key: str,
    upstash_redis_rest_url: str = "",
    upstash_redis_rest_token: str = "",
    webhook_token: Optional[str] = None,
    profiles_dir: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Path:
    """
    Convenience wrapper saving credentials into the specified or default profile's config.json.
    """
    target_pid = profile_id or get_default_profile_id()
    if not target_pid:
        raise ValueError("Cannot save credentials: No active or discovered Camoufox profile found.")
    p_dir = Path(profiles_dir).resolve() if profiles_dir else None
    return save_profile_credentials(
        profile_id=target_pid,
        username=username,
        key=key,
        upstash_redis_rest_url=upstash_redis_rest_url,
        upstash_redis_rest_token=upstash_redis_rest_token,
        webhook_token=webhook_token,
        profiles_dir=p_dir,
    )


# =========================================================================
# Camoufox Profile Tenant Discovery & Validation
# =========================================================================

def list_camoufox_profiles(profiles_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Discovers all physical Camoufox tenant profiles in profiles_dir.
    Sorts them, extracts human-friendly IDs, and checks lock statuses.
    """
    target_dir = profiles_dir or get_orchestrator_config().profiles_dir
    if not target_dir.exists() or not target_dir.is_dir():
        logger.warning(f"Profiles directory does not exist: {target_dir}")
        return []

    profiles: List[Dict[str, Any]] = []

    for entry in target_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue

        raw_name = entry.name
        # Human-friendly ID: remove 'profile_' prefix for display if present
        clean_id = raw_name[8:] if raw_name.startswith("profile_") else raw_name

        # Check lock status
        lock_file_1 = target_dir / f"{raw_name}.lock"
        lock_file_2 = entry / ".lock"
        is_locked = lock_file_1.exists() or lock_file_2.exists()

        # Gather file stats
        try:
            stat_info = entry.stat()
            mtime = stat_info.st_mtime
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = 0.0
            mtime_str = "Unknown"

        # Count items inside profile directory
        item_count = 0
        try:
            item_count = sum(1 for _ in entry.iterdir())
        except Exception:
            pass

        profiles.append({
            "id": clean_id,
            "folder_name": raw_name,
            "path": str(entry.resolve()),
            "is_locked": is_locked,
            "mtime": mtime,
            "mtime_str": mtime_str,
            "item_count": item_count,
        })

    # Sort profiles alphabetically by id
    profiles.sort(key=lambda p: p["id"].lower())
    return profiles


def verify_profile_id(profile_id: str, profiles_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Validates whether profile_id corresponds to a real folder inside profiles_dir.
    Checks both direct folder name and 'profile_' prefixed folder name.
    Strictly forbids path traversal attempts.
    """
    if not profile_id or not isinstance(profile_id, str):
        return None

    clean = profile_id.strip()
    if not clean or "/" in clean or "\\" in clean or ".." in clean:
        return None

    target_dir = (profiles_dir or get_orchestrator_config().profiles_dir).resolve()
    if not target_dir.exists():
        return None

    # Try 1: Exact folder match
    cand_1 = (target_dir / clean).resolve()
    if cand_1.exists() and cand_1.is_dir() and str(cand_1).startswith(str(target_dir)):
        return cand_1

    # Try 2: Prefixed with 'profile_'
    prefixed = f"profile_{clean}" if not clean.startswith("profile_") else clean
    cand_2 = (target_dir / prefixed).resolve()
    if cand_2.exists() and cand_2.is_dir() and str(cand_2).startswith(str(target_dir)):
        return cand_2

    # Try 3: Unprefixed if clean started with 'profile_'
    if clean.startswith("profile_"):
        unprefixed = clean[8:]
        cand_3 = (target_dir / unprefixed).resolve()
        if cand_3.exists() and cand_3.is_dir() and str(cand_3).startswith(str(target_dir)):
            return cand_3

    return None
