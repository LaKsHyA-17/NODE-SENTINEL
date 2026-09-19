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
