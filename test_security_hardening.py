"""Regression tests for production authentication and authorization safeguards."""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


client = TestClient(app)


def _login(username: str, password: str) -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_protected_routes_require_a_signed_in_user(monkeypatch):
    monkeypatch.setattr(settings, "REQUIRE_AUTH", True)
    client.cookies.clear()

    response = client.get("/api/network/graph")
    assert response.status_code == 401

    token = _login("viewer", "Viewer123!")
    response = client.get("/api/network/graph", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_permissions_are_enforced_server_side(monkeypatch):
    monkeypatch.setattr(settings, "REQUIRE_AUTH", True)
    client.cookies.clear()
    token = _login("viewer", "Viewer123!")

    response = client.post("/api/ingest/demo/reset", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403

    response = client.post(
        "/api/search/face",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("probe.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 403


def test_login_cookie_is_http_only_and_same_site(monkeypatch):
    monkeypatch.setattr(settings, "SESSION_COOKIE_SECURE", True)
    client.cookies.clear()

    response = client.post("/api/auth/login", json={"username": "viewer", "password": "Viewer123!"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "secure" in cookie


def test_security_validation_rejects_weak_secret_key():
    import pytest
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "too-short"
    s.CORS_ORIGINS = ["https://node-sentinel.onrender.com"]
    s.DEMO_MODE = False
    s.USERS_FILE = "/secure/users.json"
    s.USERS_FILE_CONFIGURED = True
    s.AUDIT_FILE = "/secure/audit.json"
    s.AUDIT_FILE_CONFIGURED = True

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        s.validate_security_configuration()


def test_security_validation_rejects_wildcard_cors():
    import pytest
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "a" * 32
    s.CORS_ORIGINS = ["*"]
    s.DEMO_MODE = False
    s.USERS_FILE = "/secure/users.json"
    s.USERS_FILE_CONFIGURED = True
    s.AUDIT_FILE = "/secure/audit.json"
    s.AUDIT_FILE_CONFIGURED = True

    with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
        s.validate_security_configuration()


def test_security_validation_rejects_demo_mode_when_auth_required():
    import pytest
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "a" * 32
    s.CORS_ORIGINS = ["https://node-sentinel.onrender.com"]
    s.DEMO_MODE = True
    s.USERS_FILE = "/secure/users.json"
    s.USERS_FILE_CONFIGURED = True
    s.AUDIT_FILE = "/secure/audit.json"
    s.AUDIT_FILE_CONFIGURED = True

    with pytest.raises(RuntimeError, match="DEMO_MODE"):
        s.validate_security_configuration()


def test_security_validation_rejects_missing_users_file():
    import pytest
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "a" * 32
    s.CORS_ORIGINS = ["https://node-sentinel.onrender.com"]
    s.DEMO_MODE = False
    s.USERS_FILE_CONFIGURED = False
    s.AUDIT_FILE = "/secure/audit.json"
    s.AUDIT_FILE_CONFIGURED = True

    with pytest.raises(RuntimeError, match="USERS_FILE"):
        s.validate_security_configuration()


def test_security_validation_rejects_bundled_users_file():
    import pytest
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "a" * 32
    s.CORS_ORIGINS = ["https://node-sentinel.onrender.com"]
    s.DEMO_MODE = False
    s.USERS_FILE = s._DEFAULT_USERS_FILE
    s.USERS_FILE_CONFIGURED = True
    s.AUDIT_FILE = "/secure/audit.json"
    s.AUDIT_FILE_CONFIGURED = True

    with pytest.raises(RuntimeError, match="bundled demo user store"):
        s.validate_security_configuration()


def test_security_validation_rejects_missing_audit_file():
    import pytest
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "a" * 32
    s.CORS_ORIGINS = ["https://node-sentinel.onrender.com"]
    s.DEMO_MODE = False
    s.USERS_FILE = "/secure/users.json"
    s.USERS_FILE_CONFIGURED = True
    s.AUDIT_FILE_CONFIGURED = False

    with pytest.raises(RuntimeError, match="AUDIT_FILE"):
        s.validate_security_configuration()


def test_security_validation_accepts_valid_ephemeral_demo_configuration():
    from app.config import Settings
    s = Settings()
    s.REQUIRE_AUTH = True
    s.SECRET_KEY = "a" * 32
    s.CORS_ORIGINS = ["https://node-sentinel.onrender.com"]
    s.DEMO_MODE = False
    s.ALLOW_EPHEMERAL_AUTH_STORAGE = True
    s.USERS_FILE = "/app/data/users.json"
    s.USERS_FILE_CONFIGURED = True
    s.AUDIT_FILE = "/app/data/audit_log.json"
    s.AUDIT_FILE_CONFIGURED = True

    # Must pass validation without raising
    s.validate_security_configuration()
    assert s.REQUIRE_AUTH is True
    assert s.DEMO_MODE is False
    assert s.ALLOW_EPHEMERAL_AUTH_STORAGE is True

