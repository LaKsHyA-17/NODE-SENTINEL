import os
from pathlib import Path


class Settings:
    PROJECT_NAME = "NODE SENTINEL"
    API_V1_STR = "/api"
    BASE_DIR = str(Path(__file__).resolve().parent.parent)

    # Security & Auth Settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "nodesentinel_insecure_dev_key_change_in_prod_99f81a")
    AUTH_TOKEN_EXPIRE_MINUTES = int(os.environ.get("AUTH_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours
    USERS_FILE = os.environ.get("USERS_FILE", str(Path(BASE_DIR) / "sample_data" / "users.json"))
    AUDIT_FILE = os.environ.get("AUDIT_FILE", str(Path(BASE_DIR) / "sample_data" / "audit_log.json"))
    CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
    # Keep-Alive Settings (Render / Cloud hosting free-tier prevention)
    KEEP_ALIVE_URL = os.environ.get("KEEP_ALIVE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL")
    KEEP_ALIVE_INTERVAL_SECONDS = int(os.environ.get("KEEP_ALIVE_INTERVAL_SECONDS", "600"))  # 10 minutes (prevents 15m idle shutdown)

    # OCR & Tesseract Settings
    TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
    OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() in ("true", "1", "yes")


settings = Settings()

