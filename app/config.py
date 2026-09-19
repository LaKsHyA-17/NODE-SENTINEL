import os
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Parse environment boolean values consistently."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    PROJECT_NAME = "NODE SENTINEL"
    API_V1_STR = "/api"
    BASE_DIR = str(Path(__file__).resolve().parent.parent)

    # A demo can be convenient locally, but it must never become the implicit
    # mode for a public deployment handling investigative information.
    APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
    IS_PRODUCTION = APP_ENV in {"production", "prod"}
    DEMO_MODE = _as_bool(os.environ.get("DEMO_MODE"), default=not IS_PRODUCTION)
    REQUIRE_AUTH = _as_bool(os.environ.get("REQUIRE_AUTH"), default=not DEMO_MODE)

    # Security & Auth Settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "")
    AUTH_TOKEN_EXPIRE_MINUTES = int(os.environ.get("AUTH_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours
    _DEFAULT_USERS_FILE = str(Path(BASE_DIR) / "sample_data" / "users.json")
    USERS_FILE = os.environ.get("USERS_FILE", _DEFAULT_USERS_FILE)
    USERS_FILE_CONFIGURED = bool(os.environ.get("USERS_FILE"))
    AUDIT_FILE = os.environ.get("AUDIT_FILE", str(Path(BASE_DIR) / "sample_data" / "audit_log.json"))
    AUDIT_FILE_CONFIGURED = bool(os.environ.get("AUDIT_FILE"))
    CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    SESSION_COOKIE_SECURE = _as_bool(os.environ.get("SESSION_COOKIE_SECURE"), default=IS_PRODUCTION)
    # Keep-Alive Settings (Render / Cloud hosting free-tier prevention)
    KEEP_ALIVE_URL = os.environ.get("KEEP_ALIVE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL")
    KEEP_ALIVE_INTERVAL_SECONDS = int(os.environ.get("KEEP_ALIVE_INTERVAL_SECONDS", "600"))  # 10 minutes (prevents 15m idle shutdown)

    # OCR & Tesseract Settings
    TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
    OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() in ("true", "1", "yes")

    def validate_security_configuration(self) -> None:
        """Fail closed when a production deployment is missing core safeguards."""
        if not self.REQUIRE_AUTH:
            return
        if len(self.SECRET_KEY) < 32:
            raise RuntimeError("SECRET_KEY must be set to a random value of at least 32 characters when REQUIRE_AUTH=true")
        if self.CORS_ORIGINS == ["*"]:
            raise RuntimeError("CORS_ORIGINS must list approved origins when REQUIRE_AUTH=true")
        if self.DEMO_MODE:
            raise RuntimeError("DEMO_MODE must be false when REQUIRE_AUTH=true")
        if not self.USERS_FILE_CONFIGURED:
            raise RuntimeError("USERS_FILE must point to a persistent, access-controlled user store when REQUIRE_AUTH=true")
        if Path(self.USERS_FILE).resolve() == Path(self._DEFAULT_USERS_FILE).resolve():
            raise RuntimeError("USERS_FILE cannot use the bundled demo user store when REQUIRE_AUTH=true")
        if not self.AUDIT_FILE_CONFIGURED:
            raise RuntimeError("AUDIT_FILE must point to persistent audit storage when REQUIRE_AUTH=true")


settings = Settings()
settings.validate_security_configuration()
