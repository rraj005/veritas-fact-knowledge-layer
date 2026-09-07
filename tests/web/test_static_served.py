"""Smoke tests for Task 16: static web UI served from /."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_with_web(tmp_path: Path):
    """Build a test FastAPI app using fakes and a temp web dir.

    The web dir is created inside tmp_path so index.html's presence activates
    the StaticFiles mount (mirroring the guard in build_app).
    """
    import json

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.llm.client import FakeLLMClient
    from veritas.pipeline import Pipeline
    from veritas.store.db import Store
    from veritas.store.embeddings import FakeEmbedder
    from veritas.store.vectors import VectorIndex

    # Point the static file server at our temp web dir.
    web_dir = tmp_path / "web"
    web_dir.mkdir()

    # Minimal index.html to satisfy the StaticFiles mount.
    (web_dir / "index.html").write_text(
        "<!doctype html><html><head><title>Veritas</title></head>"
        "<body><div id='app'></div></body></html>",
        encoding="utf-8",
    )

    store = Store(tmp_path / "veritas.db")
    store.init_schema()

    index = VectorIndex(tmp_path / "chroma", FakeEmbedder())

    def _respond(system: str, user: str) -> str:
        return json.dumps([])

    llm = FakeLLMClient(_respond)
    settings = get_settings()

    import dataclasses

    settings = dataclasses.replace(
        settings,
        upload_dir=tmp_path / "uploads",
    )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=store, index=index, llm=llm, settings=settings)

    # Monkey-patch the _WEB_DIR used inside build_app to our temp web dir.
    import veritas.api.app as _app_mod

    original_web_dir = _app_mod._WEB_DIR
    _app_mod._WEB_DIR = web_dir
    try:
        app = build_app(store=store, index=index, pipeline=pipeline, llm=llm)
    finally:
        _app_mod._WEB_DIR = original_web_dir

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def web_client(tmp_path: Path):
    """TestClient with the static web dir mounted."""
    from fastapi.testclient import TestClient

    app = _make_app_with_web(tmp_path)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_returns_html(web_client) -> None:
    """GET / must return 200 with a text/html content-type."""
    r = web_client.get("/")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    ct = r.headers.get("content-type", "")
    assert "text/html" in ct, f"Expected text/html, got {ct!r}"


def test_root_contains_veritas(web_client) -> None:
    """The served index.html must mention 'Veritas' in its markup."""
    r = web_client.get("/")
    assert "Veritas" in r.text


def test_web_dir_mounted_when_present(tmp_path: Path) -> None:
    """Creating src/veritas/web/index.html activates the static mount."""
    from fastapi.testclient import TestClient

    app = _make_app_with_web(tmp_path)
    # Verify the static route is registered (StaticFiles mount at "/")
    routes = [getattr(r, "path", None) for r in app.routes]
    # The static mount uses the root path "/" — confirm a route exists there
    assert "/" in routes or any("/" in str(r) for r in routes), "No static mount found on '/'"
    with TestClient(app) as c:
        assert c.get("/").status_code == 200


def test_api_still_reachable_alongside_static(web_client) -> None:
    """API endpoints remain reachable when static files are mounted."""
    r = web_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
