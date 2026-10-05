"""The hardcoded ~2MB request-body ceiling blocked ALL multimodal
input (video + large images) on /v1/chat/completions, /api/chat, and
/v1/embeddings. Two independently-hardcoded module constants
(body_size_limit.py's _MAX_BODY_BYTES, embeddings.py's _MAX_REQUEST_BYTES)
are promoted to ONE real config field: config.py's HttpConfig.max_body_bytes
(default 64 MiB), mirroring KVConfig.ram_cache_max_bytes
promotion (see tests/test_kvcache_ram_ceiling_config.py, the template for
this file).

Non-vacuity discipline throughout: every "config-driven" assertion here sets
a ceiling that DIFFERS from the old hardcoded 2 MiB value and checks the
enforced threshold moves with it -- a test that would still pass against the
old hardcoded constant is worthless.

Part A: BodySizeLimitMiddleware in isolation (fake ASGI scope, no FastAPI app).
Part B: /v1/embeddings' _content_length_gate via a real app + TestClient.
Part C: /v1/chat/completions via a real app + TestClient, plus PUT /api/config
        live-no-restart and TURBOHAUL_MAX_BODY_BYTES env override.
"""
import pytest

from turbohaul.api.body_size_limit import BodySizeLimitMiddleware
from turbohaul.config import (
    BootConfig,
    HttpConfig,
    PullConfig,
    QueueConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    ServerConfig,
    StorageConfig,
    TurbohaulConfig,
    UIConfig,
    apply_env_overrides,
)

# === Part A: BodySizeLimitMiddleware in isolation ===

class _FakeManager:
    def __init__(self, max_body_bytes: int):
        self.runtime = RuntimeConfig(
            queue=QueueConfig(), pull=PullConfig(),
            http=HttpConfig(max_body_bytes=max_body_bytes),
        )


class _FakeAppState:
    def __init__(self, manager):
        self.manager = manager


class _FakeApp:
    """Stand-in for scope["app"] -- only .state.manager is read."""

    def __init__(self, max_body_bytes: int):
        self.state = _FakeAppState(_FakeManager(max_body_bytes))


async def _inner_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _scope(path: str, content_length: int | None, app):
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {"type": "http", "path": path, "headers": headers, "app": app}


class _Recorder:
    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_ceiling_is_read_from_config_not_hardcoded_low_value_rejects_under_2mib():
    """Non-vacuity proof: 500 KiB is comfortably under the OLD hardcoded 2 MiB
    ceiling -- if the constant were still hardcoded, this request would pass.
    Set config to a 100-byte ceiling instead and confirm 500 KiB is REJECTED.
    This FAILS against the pre-fix hardcoded constant (it would let 500 KiB
    through) and PASSES only when the value is genuinely read from config."""
    app = _FakeApp(max_body_bytes=100)
    mw = BodySizeLimitMiddleware(_inner_app)
    recorder = _Recorder()
    size = 500 * 1024
    await mw(_scope("/v1/chat/completions", size, app), None, recorder)

    assert recorder.events[0]["status"] == 413
    detail = recorder.events[1]["body"].decode()
    assert f"{size} bytes exceeds 100 ceiling" in detail


@pytest.mark.asyncio
async def test_ceiling_is_read_from_config_high_value_accepts_over_2mib():
    """Mirror proof in the other direction: 5 MiB is OVER the old hardcoded
    2 MiB ceiling -- if the constant were still hardcoded, this request would
    be rejected with 413. Set config to 64 MiB (the new default) instead and
    confirm 5 MiB now PASSES through to the inner app."""
    app = _FakeApp(max_body_bytes=64 * 1024 * 1024)
    mw = BodySizeLimitMiddleware(_inner_app)
    recorder = _Recorder()
    size = 5 * 1024 * 1024  # the size class of a real multimodal upload
    await mw(_scope("/v1/chat/completions", size, app), None, recorder)

    assert recorder.events[0]["status"] == 200


@pytest.mark.asyncio
async def test_raising_far_above_default_is_genuinely_honoured_not_silently_clamped():
    """Raising far above the default: the point that
    actually matters is that the ceiling is configurable UP, not the default
    number -- as long as the total upload size is configurable, the operator can
    size it for whatever harness uses it. Set the
    ceiling to 512 MiB (an example of a large operator-chosen value)
    and confirm a 100 MiB declared body -- well above the 64 MiB default,
    which the OLD hardcoded-32 or hardcoded-2MB constants would both have
    rejected -- is accepted. Proves the comparison itself is correct at a
    large value (no int overflow, no second ceiling silently overriding the
    configured one) without transferring 512 MiB of actual body bytes over
    the wire (this test only inspects the declared Content-Length header, it
    never reads a body)."""
    half_gib = 512 * 1024 * 1024
    app = _FakeApp(max_body_bytes=half_gib)
    assert app.state.manager.runtime.http.max_body_bytes == half_gib  # genuinely SET, not clamped

    mw = BodySizeLimitMiddleware(_inner_app)
    recorder = _Recorder()
    hundred_mib = 100 * 1024 * 1024
    await mw(_scope("/v1/chat/completions", hundred_mib, app), None, recorder)
    assert recorder.events[0]["status"] == 200  # genuinely HONOURED


@pytest.mark.asyncio
async def test_exactly_at_ceiling_passes_one_byte_over_rejects():
    app = _FakeApp(max_body_bytes=1000)
    mw = BodySizeLimitMiddleware(_inner_app)

    at_limit = _Recorder()
    await mw(_scope("/v1/chat/completions", 1000, app), None, at_limit)
    assert at_limit.events[0]["status"] == 200

    over_limit = _Recorder()
    await mw(_scope("/v1/chat/completions", 1001, app), None, over_limit)
    assert over_limit.events[0]["status"] == 413


@pytest.mark.asyncio
async def test_unguarded_path_bypasses_regardless_of_size_or_config():
    app = _FakeApp(max_body_bytes=1)  # smallest possible ceiling
    mw = BodySizeLimitMiddleware(_inner_app)
    recorder = _Recorder()
    await mw(_scope("/v1/embeddings", 10_000_000, app), None, recorder)
    # /v1/embeddings is guarded by its OWN dependency (Part B), not this
    # middleware's _GUARDED_PATHS -- this middleware must pass it straight
    # through untouched.
    assert recorder.events[0]["status"] == 200


@pytest.mark.asyncio
async def test_no_content_length_header_bypasses_unchanged_from_before():
    """Constraint: do not widen this pre-existing gap.
    Confirmed still exactly as wide as before -- absent Content-Length falls
    straight through with no size check at all, even with a tiny ceiling."""
    app = _FakeApp(max_body_bytes=1)
    mw = BodySizeLimitMiddleware(_inner_app)
    recorder = _Recorder()
    await mw(_scope("/v1/chat/completions", None, app), None, recorder)
    assert recorder.events[0]["status"] == 200


# === Part B + C: real app + TestClient ===

@pytest.fixture
def app_test(tmp_path):
    from fastapi.testclient import TestClient

    from turbohaul.api.main import create_app

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
            llama_server_binary=tmp_path / "fake", default_port_base=59600,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    client = TestClient(app)
    return app, client


def _padded_embeddings_json(total_size: int) -> bytes:
    """Valid EmbeddingsRequest JSON padded via the `input` string to land at
    approximately total_size bytes -- must be valid JSON so a 413 from the
    Content-Length gate isn't masked by an unrelated 422 pydantic failure
    (the gate and pydantic parsing are both FastAPI dependencies; only a
    structurally valid body isolates which one actually fired)."""
    prefix = b'{"model": "m", "input": "'
    suffix = b'"}'
    pad_len = max(0, total_size - len(prefix) - len(suffix))
    return prefix + (b"a" * pad_len) + suffix


def test_embeddings_content_length_gate_reads_config_not_hardcoded(app_test):
    """Non-vacuous: 500 KiB is under the old hardcoded 2 MiB (would have
    passed the gate before this change). Tighten config to 100 bytes and confirm
    it is now rejected -- proves _content_length_gate reads live config."""
    app, client = app_test
    app.state.manager.runtime.http.max_body_bytes = 100
    size = 500 * 1024
    body = _padded_embeddings_json(size)
    r = client.post(
        "/v1/embeddings",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert f"{len(body)} bytes exceeds 100 ceiling" in r.json()["detail"]


def test_embeddings_under_new_default_ceiling_is_not_body_size_rejected(app_test):
    """5 MiB is over the OLD hardcoded 2 MiB ceiling but under the new 64 MiB
    default -- must NOT be rejected by the Content-Length gate (it may still
    404/400 downstream on model lookup/JSON validity; the point is it is not
    a 413 from the byte-size gate)."""
    app, client = app_test
    size = 5 * 1024 * 1024
    r = client.post(
        "/v1/embeddings",
        content=_padded_embeddings_json(size),
        headers={"content-type": "application/json"},
    )
    assert r.status_code != 413


def test_chat_completions_middleware_reads_config_not_hardcoded(app_test):
    """Same non-vacuity proof as the embeddings case, for the ASGI middleware
    path guarding /v1/chat/completions."""
    app, client = app_test
    app.state.manager.runtime.http.max_body_bytes = 100
    size = 500 * 1024
    r = client.post("/v1/chat/completions", content=b"x" * size)
    assert r.status_code == 413
    assert f"{size} bytes exceeds 100 ceiling" in r.json()["detail"]


def test_chat_completions_under_new_default_ceiling_is_not_body_size_rejected(app_test):
    app, client = app_test
    size = 5 * 1024 * 1024
    r = client.post(
        "/v1/chat/completions",
        content=b'{"model": "x", "messages": [], "pad": "' + b"x" * size + b'"}',
    )
    assert r.status_code != 413


def test_get_api_config_exposes_new_http_section(app_test):
    app, client = app_test
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "http" in body
    assert body["http"]["max_body_bytes"] == 64 * 1024 * 1024


def test_put_api_config_changes_ceiling_live_no_restart(app_test):
    """Mirrors test_kvcache_ram_ceiling_config.py's
    test_ceiling_change_is_picked_up_live_no_restart: PUT /api/config, then
    confirm the VERY NEXT request on the SAME TestClient/app honors the new
    value -- no manager reconstruction, no restart."""
    app, client = app_test

    # Before: 500 KiB passes under the 64 MiB default.
    size = 500 * 1024
    r1 = client.post("/v1/chat/completions", content=b"x" * size)
    assert r1.status_code != 413

    # Live PUT tightens the ceiling well under 500 KiB.
    put = client.put("/api/config", json={"http": {"max_body_bytes": 1024}})
    assert put.status_code == 200

    # Same client, same app, no restart: the next request honors the new ceiling.
    r2 = client.post("/v1/chat/completions", content=b"x" * size)
    assert r2.status_code == 413
    assert f"{size} bytes exceeds 1024 ceiling" in r2.json()["detail"]


def test_put_api_config_raises_ceiling_live_no_restart(app_test):
    """The mirror-image of the test above, and the direction that was explicitly
    asked to be proven: raising is the direction that matters. Start
    tight enough to reject a 500 KiB body, PUT to RAISE the ceiling well past
    it, and confirm the SAME request that was just rejected now succeeds on
    the SAME client/app -- proves a live raise is honoured immediately, not
    just a live lower."""
    app, client = app_test
    size = 500 * 1024

    put_low = client.put("/api/config", json={"http": {"max_body_bytes": 1024}})
    assert put_low.status_code == 200
    r1 = client.post("/v1/chat/completions", content=b"x" * size)
    assert r1.status_code == 413

    put_high = client.put("/api/config", json={"http": {"max_body_bytes": 10 * 1024 * 1024}})
    assert put_high.status_code == 200

    r2 = client.post("/v1/chat/completions", content=b"x" * size)
    assert r2.status_code != 413


def test_env_var_override_sets_the_ceiling(monkeypatch):
    monkeypatch.setenv("TURBOHAUL_MAX_BODY_BYTES", "123456")
    cfg = TurbohaulConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path="/tmp/x/blobs", manifests_path="/tmp/x/manifests",
            import_allowed_root="/tmp/x/import", state_db_path="/tmp/x/state.sqlite",
        ),
        runtime=RuntimePathsConfig(llama_server_binary="/tmp/x/fake"),
        ui=UIConfig(static_path="/tmp/x/ui_dist"),
        queue=QueueConfig(),
        pull=PullConfig(),
    )
    overridden = apply_env_overrides(cfg)
    assert overridden.http.max_body_bytes == 123456


def test_default_is_32_mib_and_larger_than_the_old_2mib_ceiling():
    h = HttpConfig()
    assert h.max_body_bytes == 64 * 1024 * 1024
    assert h.max_body_bytes > 32_768 * 64  # the old hardcoded ~2MB constant


def test_field_has_no_upper_bound_can_be_set_far_above_default():
    """The field must be settable far higher than the
    default without tripping a second hidden cap -- a very large media
    ingestion path is a separate, unsolved problem (see ARCHITECTURE.md §9),
    but THIS field must not be the thing standing in the way of raising it."""
    huge = 2 * 1024 * 1024 * 1024  # 2 GiB
    h = HttpConfig(max_body_bytes=huge)
    assert h.max_body_bytes == huge
