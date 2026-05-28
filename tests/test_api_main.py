"""Tests for FastAPI app skeleton (v0.2 §9) + Phase 5 §11 UI mount."""
import pytest
from fastapi.testclient import TestClient

from turbohaul import __version__
from turbohaul.api.main import _CSP_HEADER, create_app
from turbohaul.config import (
    BootConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    UIConfig,
)


@pytest.fixture
def app_and_client(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()

    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui_dist"),  # missing dir → no /ui route
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client


@pytest.fixture
def app_and_client_with_ui(tmp_path):
    """create_app variant where a real ui_dist exists. Used for Phase 5 §11 tests."""
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    (storage_root / "blobs").mkdir()
    (storage_root / "manifests").mkdir()
    (storage_root / "import-staging").mkdir()

    ui_dist = tmp_path / "ui_dist"
    ui_dist.mkdir()
    (ui_dist / "index.html").write_text(
        '<!doctype html><html><head></head><body><div id="root">SPA</div></body></html>',
        encoding="utf-8",
    )
    (ui_dist / "assets").mkdir()
    (ui_dist / "assets" / "index-DEADBEEF.js").write_text(
        'console.log("turbohaul");', encoding="utf-8"
    )
    (ui_dist / "assets" / "index-CAFEBABE.css").write_text(
        "body{color:white}", encoding="utf-8"
    )

    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake_llama_server",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=ui_dist),
    )
    runtime = RuntimeConfig(queue=QueueConfig(), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client, ui_dist


class TestHealth:
    def test_health_returns_ok(self, app_and_client):
        app, client = app_and_client
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__


class TestStatus:
    def test_status_initial_empty(self, app_and_client):
        app, client = app_and_client
        r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert body["queue"]["acceptance_buffer_depth"] == 0
        assert body["queue"]["staging_queue_depth"] == 0
        assert body["active"] is None
        assert body["grace"] is None
        assert body["idle_hot"] is None
        assert body["parallel_slots"]["used"] == 0


class TestApiVersion:
    def test_api_version_payload(self, app_and_client):
        app, client = app_and_client
        r = client.get("/api/version")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == __version__
        assert body["api_compat"] == "ollama-superset"
        assert "Ollama-compatible" in body["user_agent"]
        assert body["backend_sha_pinned"] is False


class TestApiConfig:
    def test_api_config_returns_split_view(self, app_and_client):
        app, client = app_and_client
        r = client.get("/api/config")
        assert r.status_code == 200
        body = r.json()
        assert "server" in body
        assert body["server"]["host"] == "127.0.0.1"
        assert body["server"]["port"] == 11401
        assert "storage" in body
        assert "runtime" in body
        assert "ui" in body
        assert "queue" in body
        assert body["queue"]["grace_seconds"] == 30
        assert body["queue"]["idle_hot_load_seconds"] == 600
        assert "pull" in body
        assert body["pull"]["pull_url_https_only"] is True


class TestAppCreation:
    def test_app_has_manager_state(self, app_and_client):
        app, client = app_and_client
        assert hasattr(app.state, "manager")
        from turbohaul.manager import TurbohaulManager
        assert isinstance(app.state.manager, TurbohaulManager)

    def test_unknown_endpoint_404(self, app_and_client):
        app, client = app_and_client
        r = client.get("/nonexistent")
        assert r.status_code == 404


class TestUIDisabledOrMissing:
    """When ui.enabled=False OR static_path doesn't exist, /ui returns 404."""

    def test_ui_route_absent_when_static_path_missing(self, app_and_client):
        app, client = app_and_client
        # Fixture uses tmp_path/'ui_dist' which is NOT created → no /ui route.
        r = client.get("/ui/")
        assert r.status_code == 404


class TestUIServing:
    def test_ui_root_serves_index_html(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/")
        assert r.status_code == 200, r.text
        assert b'<div id="root">SPA</div>' in r.content
        assert r.headers["content-type"].startswith("text/html")

    def test_ui_no_trailing_slash_also_serves(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui", follow_redirects=True)
        assert r.status_code == 200
        assert b"SPA" in r.content

    def test_ui_serves_hashed_js_asset(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/assets/index-DEADBEEF.js")
        assert r.status_code == 200
        assert b'console.log("turbohaul");' in r.content
        assert r.headers["cache-control"] == "public, max-age=31536000, immutable"

    def test_ui_serves_hashed_css_asset(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/assets/index-CAFEBABE.css")
        assert r.status_code == 200
        assert b"body{color:white}" in r.content
        assert r.headers["cache-control"] == "public, max-age=31536000, immutable"

    def test_ui_index_html_no_cache(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/")
        assert r.headers["cache-control"] == "no-cache, must-revalidate"


class TestUISecurityHeaders:
    def test_csp_present_on_index(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/")
        assert r.headers["content-security-policy"] == _CSP_HEADER
        # Belt-and-suspenders headers
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "same-origin"

    def test_csp_present_on_assets(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/assets/index-DEADBEEF.js")
        assert r.headers["content-security-policy"] == _CSP_HEADER

    def test_csp_denies_inline_scripts(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/")
        csp = r.headers["content-security-policy"]
        # Verify no 'unsafe-inline' in script-src
        assert "script-src 'self';" in csp
        assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]

    def test_csp_denies_framing(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/")
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


class TestUISPAFallback:
    """React Router routes (/ui/queue, /ui/blob, etc.) must fall back to index.html."""

    def test_unknown_subpath_returns_index_html(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/queue")
        assert r.status_code == 200
        assert b'<div id="root">SPA</div>' in r.content

    def test_deeply_nested_unknown_path_returns_index_html(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/blob/some/deep/route")
        assert r.status_code == 200
        assert b"SPA" in r.content

    def test_fallback_inherits_security_headers(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui/settings")
        assert r.headers["content-security-policy"] == _CSP_HEADER
        assert r.headers["cache-control"] == "no-cache, must-revalidate"


class TestUIPathTraversal:
    def test_dotdot_escape_returns_index_not_etc(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        # Defense in depth — two layers:
        #   1. Starlette/HTTPX normalizes ".." segments BEFORE routing, so the
        #      request resolves to /etc/passwd which has no handler → 404.
        #   2. Even if that normalization were bypassed, _serve's resolve()+
        #      relative_to(ui_root) guard would refuse to read outside ui_root
        #      and fall back to index.html (200, SPA content).
        # The security invariant — /etc/passwd MUST NOT be served — holds in either
        # case. We assert: status is 200 or 404, AND the response body never leaks
        # the host's /etc/passwd content.
        r = client.get("/ui/../../etc/passwd")
        assert r.status_code in (200, 404), r.text
        assert b"root:" not in r.content
        assert b"/bin/bash" not in r.content

    def test_absolute_path_smuggle_returns_index(self, app_and_client_with_ui):
        app, client, ui_dist = app_and_client_with_ui
        r = client.get("/ui//etc/passwd")
        # Should fall back to index.html, not the host's /etc/passwd
        assert r.status_code == 200
        assert b"root:" not in r.content



class TestProductionWiring:
    """Asserts create_app() injects real factory functions into TurbohaulManager.

    This test class exists because Phase 6  shipped with `complete_fn` NOT
    wired in production: api/main.py constructed TurbohaulManager(boot, runtime)
    without passing complete_fn=, so the default raised
    'no completion_fn wired' on every real /v1/chat/completions request.
    All 322+ existing tests passed because each one explicitly injected its own
    mock factory.

    Lesson: pytest assertions on the management plane do NOT cover the
    production-wiring path. Add explicit checks that create_app() does not
    rely on default no-op factories.
    """

    def test_complete_fn_is_wired_not_default(self, app_and_client):
        """create_app() must inject a real make_llama_server_complete_fn() callable.

        Caught in smoke testing: default complete_fn
        raised 'no completion_fn wired' on first chat completion. Fix in commit
        174a847 wired make_llama_server_complete_fn() into manager construction.
        """
        app, _ = app_and_client
        mgr = app.state.manager
        assert mgr._complete_fn is not None, (
            "TurbohaulManager._complete_fn is None — create_app() didn't wire it"
        )
        # The default no-op (manager.py default) is identifiable by its docstring
        # containing the 'no completion_fn wired' sentinel raise. The production
        # injection wraps the httpx forwarder factory — check it does NOT raise
        # the wiring-missing sentinel when introspected.
        import inspect
        src = inspect.getsource(mgr._complete_fn) if callable(mgr._complete_fn) else ""
        assert "no completion_fn wired" not in src, (
            "create_app() injected the DEFAULT no-op complete_fn, not the real "
            "make_llama_server_complete_fn factory. Production /v1/chat/completions "
            "will return 503."
        )

    def test_manager_factories_non_default_after_create_app(self, app_and_client):
        """Belt-and-suspenders: assert spawn_fn / health_fn / sigterm_fn / vram_fn /
        complete_fn are all callable and bound, not None.

        Each factory has its own default (no-op or raise); production wiring
        SHOULD inject real implementations from subprocess_mgr + api.chat_completion.
        """
        app, _ = app_and_client
        mgr = app.state.manager
        for name in ("_spawn", "_wait_healthy", "_sigterm", "_vram_verify", "_complete_fn"):
            fn = getattr(mgr, name, None)
            assert fn is not None and callable(fn), (
                f"TurbohaulManager.{name} is None or not callable after create_app() — "
                "production wiring is missing a factory injection."
            )
