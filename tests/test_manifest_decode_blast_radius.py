"""One unreadable file must not 500 the API — three triggers.

`read_manifest` (manifest.py) does `yaml.safe_load(path.read_text())` in no try
and raises SIX classes from three libraries plus the stdlib. Four consumers each
hand-copied their own enumeration of that surface, and the copies diverged:
**no two agreed** -- each stopped at a different class of the surface:

    api/models.py     1st copy   FileNotFound/ManifestVal/YAMLError/ValidationError
    api/ollama.py     2nd copy   (same four)
    api/manifests.py  3rd copy   ...+ValidationError, but never YAMLError
    api/plugins.py    4th copy   (the same four again, "four ways")

Three triggers came out of that, and the third is the one that matters most
because it is the most ordinary:

  A  a manifest that is not valid UTF-8 -> 5/5 routes 500.
     `UnicodeDecodeError` is a **ValueError, not an OSError**, so every
     `except OSError` around a `read_text()` — which reads as "unreadable
     file" — sails straight past it.
  C  a manifest with a plain YAML TYPO -> /api/manifests and /{tag} 500.
     No encoding exotica at all, and it reproduces the manifests-route
     failure, on the very line whose own comment says it was already handled.
  B  a model_meta.json that is not valid UTF-8 -> GET /api/blobs 500, PUT
     description 500, and DELETE /api/delete 500 **after the blob has already
     been unlinked** — a 500 that means "succeeded". `read_model_meta`'s
     docstring promised "NEVER raises" and did not keep it.

⛔ THE NO-REGRESSION HALF IS AS IMPORTANT AS THE FIX. /v1/models, /api/tags and
/api/plugins ALREADY survive trigger C — they log "skipping unreadable manifest
… ParserError" and keep serving their other models. Over-refusal is as much a
bug as under-refusal, so the tests below assert they still SERVE THE HEALTHY
SIBLING, not merely that they avoid a 500.
"""
import json

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

import turbohaul.api.chat_completion as chat_completion_mod
import turbohaul.api.embeddings as embeddings_mod
import turbohaul.api.manifests as manifests_mod
import turbohaul.api.models as models_mod
import turbohaul.api.ollama as ollama_mod
import turbohaul.api.plugins as plugins_mod
import turbohaul.model_meta as model_meta_mod
from turbohaul.api.import_ import GGUF_MAGIC
from turbohaul.api.main import create_app
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
from turbohaul.manifest import ManifestValidationError, read_manifest


UNDECODABLE = b"\xff\xfe not valid utf-8 at all\x00"
YAML_TYPO = b"model_tag: bad-model\nllama_server_flags: {unclosed: [1, 2\n"
SAMPLE_SHA = "abcdef01" + "0" * 56

# This is NOT a list of every route that reads a manifest through
# read_manifest. Every entry below is DISCOVERY or
# MANAGEMENT (a GET with no request body, seeded via _seed/_plant); zero are
# INFERENCE routes, and that gap is exactly why the crash could survive here undetected
# A comment that overstates its own guard re-arms the
# trap for the next reader -- see TestFourInferenceRoutesAnswerDeliberately
# below for the four it was missing (openai chat, ollama chat, embeddings,
# plugin invoke), which cannot share this GET-based ROUTES/DEGRADE_AND_SERVE
# shape: they are POST, take a request body, and -- being single-tag routes,
# not listings -- answer a bad manifest with a deliberate 500, not a skip.
# The bare /models path is registered on the same handler function as
# /v1/models, so it carries the same coverage here.
ROUTES = ["/api/manifests", "/api/manifests/bad-model", "/v1/models", "/models",
          "/api/tags", "/api/plugins"]
# The three that must keep SERVING, not merely keep off 500, under a bad entry.
# Each is paired with the healthy tag it is actually supposed to still list:
# /api/plugins enumerates PLUGIN manifests, so a healthy MODEL is invisible to
# it and asserting otherwise would be a probe aimed at a shape it cannot
# observe -- the same defect class this whole bug is made of. Such a probe is
# caught only by failing on correct code.
DEGRADE_AND_SERVE = [
    ("/v1/models", "good-model"),
    ("/api/tags", "good-model"),
    ("/api/plugins", "good-plugin"),
]


@pytest.fixture
def app_test(tmp_path):
    storage_root = tmp_path / "state"
    storage_root.mkdir()
    for d in ("blobs", "manifests", "import-staging"):
        (storage_root / d).mkdir()
    boot = BootConfig(
        server=ServerConfig(),
        storage=StorageConfig(
            blob_store_path=storage_root / "blobs",
            manifests_path=storage_root / "manifests",
            import_allowed_root=storage_root / "import-staging",
            state_db_path=storage_root / "state.sqlite",
        ),
        runtime=RuntimePathsConfig(
            llama_server_binary=tmp_path / "fake", default_port_base=59500
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    app = create_app(
        boot,
        RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig()),
        auto_start_worker=False,
        auto_boot_reconcile=False,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield app, client, storage_root


def _manifests_dir(app):
    return app.state.manager.boot.storage.manifests_path


def _seed_plugin(app, tag="good-plugin"):
    """A healthy PLUGIN manifest, written through the real writer so it is
    exactly what the plugin listing expects to find."""
    from turbohaul.manifest import PluginManifest, write_manifest_atomic

    write_manifest_atomic(_manifests_dir(app), PluginManifest(
        model_tag=tag, kind="plugin", lane="cpu",
        resource_key="whisperx-main", capabilities=["transcribe"],
    ))


def _seed(client, tag):
    r = client.put(f"/api/manifests/{tag}", json={
        "model_tag": tag, "display_name": f"Model {tag}",
        "gguf_blob_sha256": SAMPLE_SHA, "gguf_size_bytes": 10_000_000_000,
        "context_size": 4096, "expected_vram_bytes": 11_000_000_000,
        "llama_server_flags": {"ctx_size": 4096},
    })
    assert r.status_code == 200, r.text
    return r


def _plant(app, tag, payload: bytes):
    """Write raw BYTES. Takes bytes, not str, deliberately: a `str`-typed
    helper cannot express a non-UTF-8 file at all, and a test named for a
    shape its own fixture cannot build is the defect this bug is made of."""
    (_manifests_dir(app) / f"{tag}.yaml").write_bytes(payload)


def _import_blob(client, storage_root, body=b"blobdata"):
    src = storage_root / "import-staging" / "probe.gguf"
    src.write_bytes(GGUF_MAGIC + body)
    r = client.post("/api/import", json={"path": str(src)})
    assert r.status_code == 200, r.text
    return r.json()["sha256"]


# ===================================================================
# The drift guard. Its expectation is DERIVED from the producer, not
# hardcoded — a gate whose "expected" half is a literal can only ever
# agree with itself.
# ===================================================================

def _observe_producer_failure_classes(tmp_path):
    """Provoke read_manifest into every failure it has and RECORD what came
    out. This is the contract the four consumers must cover, measured rather
    than asserted."""
    root = tmp_path / "observe"
    root.mkdir()
    observed = set()

    def note(tag, write):
        write(root / f"{tag}.yaml")
        try:
            read_manifest(root, tag)
        except Exception as e:  # noqa: BLE001 - recording, not handling
            observed.add(type(e))
        else:
            pytest.fail(f"fixture {tag!r} was supposed to make read_manifest raise")

    # absent
    try:
        read_manifest(root, "not-there")
    except Exception as e:  # noqa: BLE001
        observed.add(type(e))
    # not valid UTF-8
    note("undecodable", lambda p: p.write_bytes(UNDECODABLE))
    # valid UTF-8, invalid YAML
    note("typo", lambda p: p.write_bytes(YAML_TYPO))
    # valid YAML whose root is not a mapping
    note("scalar", lambda p: p.write_text("just-a-string\n"))
    # valid mapping that violates the Manifest schema
    note("schema", lambda p: p.write_text("model_tag: schema\ncontext_size: 4096\n"))
    # unreadable for an OS reason: a DIRECTORY where a file is expected
    (root / "isadir.yaml").mkdir()
    try:
        read_manifest(root, "isadir")
    except Exception as e:  # noqa: BLE001
        observed.add(type(e))
    return observed


def _covers(tuple_, exc_type) -> bool:
    return issubclass(exc_type, tuple(tuple_))


CONSUMERS = [
    ("api/manifests.py", manifests_mod),
    ("api/models.py", models_mod),
    ("api/ollama.py", ollama_mod),
    ("api/plugins.py", plugins_mod),
    # The two sites that have NO _UNREADABLE_MANIFEST at all
    # -- confirmed by grep, not assumed (one might assume that
    # all four inference modules already have one; that is wrong for
    # exactly these two).
    ("api/chat_completion.py", chat_completion_mod),
    ("api/embeddings.py", embeddings_mod),
]


class TestConsumersCoverTheProducer:
    def test_the_producer_really_does_raise_a_wide_surface(self, tmp_path):
        observed = _observe_producer_failure_classes(tmp_path)
        # Not a hardcoded expectation of WHICH classes -- an assertion that
        # the surface is genuinely wide, so the coverage test below is not
        # vacuously true against a surface of one.
        assert len(observed) >= 5, f"only provoked {observed}"
        assert UnicodeDecodeError in observed, (
            "the trigger-A class must be reachable from read_manifest, or the "
            "coverage assertions below prove nothing about it"
        )
        assert any(issubclass(t, yaml.YAMLError) for t in observed), (
            "the trigger-C class must be reachable from read_manifest"
        )

    @pytest.mark.parametrize("name,mod", CONSUMERS)
    def test_each_consumer_covers_every_class_the_producer_raises(
        self, name, mod, tmp_path
    ):
        observed = _observe_producer_failure_classes(tmp_path)
        missing = sorted(
            t.__name__ for t in observed if not _covers(mod._UNREADABLE_MANIFEST, t)
        )
        assert not missing, (
            f"{name}'s _UNREADABLE_MANIFEST does not cover {missing} -- "
            f"read_manifest raises it and this consumer would 500"
        )

    def test_no_two_consumers_disagree(self):
        """The drift guard proper. Four file-local copies, kept file-local on
        purpose (the no-shared-helper constraint recorded in ollama.py), but
        held to ONE coverage by this assertion instead of by a shared import.
        A shared ASSERTION rather than shared CODE."""
        sets = {name: frozenset(mod._UNREADABLE_MANIFEST) for name, mod in CONSUMERS}
        distinct = set(sets.values())
        assert len(distinct) == 1, (
            "the four enumerations have drifted apart again: "
            + "; ".join(
                f"{n}={sorted(t.__name__ for t in s)}" for n, s in sorted(sets.items())
            )
        )


# ===================================================================
# Trigger A — a manifest that is not valid UTF-8
# ===================================================================

class TestTriggerAUndecodableManifest:
    @pytest.mark.parametrize("route", ROUTES)
    def test_route_does_not_500(self, app_test, route):
        app, client, _ = app_test
        _seed(client, "good-model")
        _seed(client, "bad-model")
        assert client.get(route).status_code == 200, "control: healthy before planting"

        _plant(app, "bad-model", UNDECODABLE)

        assert client.get(route).status_code != 500, (
            f"{route} 500s on one undecodable manifest"
        )

    @pytest.mark.parametrize("route,healthy", DEGRADE_AND_SERVE)
    def test_healthy_entries_still_served(self, app_test, route, healthy):
        app, client, _ = app_test
        _seed(client, "good-model")
        _seed_plugin(app)
        _seed(client, "bad-model")
        _plant(app, "bad-model", UNDECODABLE)

        r = client.get(route)

        assert r.status_code == 200
        assert healthy in r.text, (
            f"{route} avoided the 500 but stopped serving {healthy} too "
            "-- over-refusal is as much a bug as under-refusal"
        )

    def test_the_management_listing_reports_the_bad_row_as_unreadable(self, app_test):
        app, client, _ = app_test
        _seed(client, "good-model")
        _seed(client, "bad-model")
        _plant(app, "bad-model", UNDECODABLE)

        rows = {r["model_tag"]: r for r in client.get("/api/manifests").json()["manifests"]}

        assert rows["bad-model"]["error"] == "unreadable"
        assert rows["good-model"]["display_name"] == "Model good-model"


# ===================================================================
# Trigger C — an ordinary YAML typo
# ===================================================================

class TestTriggerCYamlTypo:
    @pytest.mark.parametrize("route", ROUTES)
    def test_route_does_not_500(self, app_test, route):
        app, client, _ = app_test
        _seed(client, "good-model")
        _seed(client, "bad-model")
        assert client.get(route).status_code == 200, "control: healthy before planting"

        _plant(app, "bad-model", YAML_TYPO)

        assert client.get(route).status_code != 500, (
            f"{route} 500s on an ordinary YAML typo -- no encoding exotica required"
        )

    @pytest.mark.parametrize("route,healthy", DEGRADE_AND_SERVE)
    def test_the_routes_that_already_survived_still_survive(self, app_test, route, healthy):
        """⛔ REGRESSION GUARD. These three handle a typo gracefully, so this
        guard stays. If widening their tuples ever makes them refuse instead of
        skip, that is a new bug, not a fix."""
        app, client, _ = app_test
        _seed(client, "good-model")
        _seed_plugin(app)
        _seed(client, "bad-model")
        _plant(app, "bad-model", YAML_TYPO)

        r = client.get(route)

        assert r.status_code == 200
        assert healthy in r.text
        assert "bad-model" not in r.text, "the unreadable entry must be skipped, not served"


# ===================================================================
# The two per-tag routes answer DIFFERENTLY, on purpose
# ===================================================================

class TestPerTagRoutesDifferDeliberately:
    def test_openai_discovery_keeps_its_500_and_it_is_now_HANDLED(self, app_test):
        """The existing design chooses a handled 5xx here: the client did nothing wrong, so
        not a 4xx, and not a 404 because that hides a corrupt manifest behind
        'you typo'd it'. UnicodeDecodeError belongs in that arm -- which
        COMPLETES that design rather than reversing it. Without it, an
        undecodable file produces an UNHANDLED 500: no fixed message, no log."""
        app, client, _ = app_test
        _seed(client, "bad-model")
        _plant(app, "bad-model", UNDECODABLE)

        r = client.get("/v1/models/bad-model")

        assert r.status_code == 500
        assert r.json()["detail"] == "manifest for 'bad-model' is present but unreadable"

    def test_openai_discovery_500_leaks_no_manifest_content(self, app_test):
        """The reason the producer is NOT wrapped. models.py's rule: 'never
        str(e): yaml.YAMLError leaks parser position/content, pydantic
        ValidationError echoes manifest field VALUES.' This route does not
        return the body on success, so its caller is not entitled to it."""
        app, client, _ = app_test
        _seed(client, "bad-model")
        _plant(app, "bad-model", b"model_tag: bad-model\nsecret_flag: [unclosed\n")

        detail = client.get("/v1/models/bad-model").json()["detail"]

        assert "secret_flag" not in detail
        assert "unclosed" not in detail
        assert "line" not in detail.lower(), "no parser position in a client-visible body"

    def test_management_per_tag_answers_400_with_the_parser_detail(self, app_test):
        """The MANAGEMENT route hands back the entire manifest on success, so
        naming the line of the operator's own typo discloses nothing they could
        not already read -- and it is what they need to fix it. Entitlement is
        the distinction between these two routes, not inconsistency."""
        app, client, _ = app_test
        _seed(client, "bad-model")
        _plant(app, "bad-model", YAML_TYPO)

        r = client.get("/api/manifests/bad-model")

        assert r.status_code == 400
        assert r.json()["detail"], "the operator gets the parser's own message"

    def test_management_per_tag_does_not_500_on_undecodable(self, app_test):
        app, client, _ = app_test
        _seed(client, "bad-model")
        _plant(app, "bad-model", UNDECODABLE)

        assert client.get("/api/manifests/bad-model").status_code == 400

    def test_absent_still_404_not_conflated_with_unreadable(self, app_test):
        """FileNotFoundError IS an OSError, so widening these arms could easily
        have swallowed 'absent' into 'unreadable'. The 404 arm must stay first."""
        app, client, _ = app_test
        assert client.get("/api/manifests/never-existed").status_code == 404
        assert client.get("/v1/models/never-existed").status_code == 404


# ===================================================================
# Trigger B — an undecodable model_meta.json
# ===================================================================

class TestTriggerBModelMeta:
    def test_read_model_meta_keeps_its_never_raises_promise(self, tmp_path):
        manifests_root = tmp_path / "manifests"
        manifests_root.mkdir()
        model_meta_mod.meta_path(manifests_root).write_bytes(UNDECODABLE)

        assert model_meta_mod.read_model_meta(manifests_root) == {}

    def test_blob_routes_do_not_500(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        assert client.get("/api/blobs").status_code == 200, "control"

        model_meta_mod.meta_path(_manifests_dir(app)).write_bytes(UNDECODABLE)

        assert client.get("/api/blobs").status_code == 200
        assert client.put(
            f"/api/blobs/{sha}/description", json={"description": "x"}
        ).status_code == 200

    def test_delete_reports_the_success_it_actually_was(self, app_test):
        """⛔ THE SEVERITY-DEFINING CASE. delete_blob unlinks, prune_digest then
        raised, and the route reported 500 for a destruction that had already
        happened -- a caller retrying gets 404, and blob_changed never fired."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        blob = storage / "blobs" / "sha256" / sha[:2] / sha
        model_meta_mod.meta_path(_manifests_dir(app)).write_bytes(UNDECODABLE)

        r = client.request("DELETE", "/api/delete", json={"sha256": sha})

        assert r.status_code == 200, "a 500 here would be a 500 that means 'succeeded'"
        assert not blob.exists(), "and the blob really was removed"


class TestCleanupCannotInvertACompletedDelete:
    """Fixing read_model_meta closes the decode INSTANCE. This closes the
    CLASS: prune_digest also WRITES, and a write fails on its own terms with
    the read perfectly healthy. Fixing the instance and leaving the class is
    a mistake to avoid; this test guards against repeating it."""

    def test_a_prune_WRITE_failure_still_reports_success_and_publishes(
        self, app_test, monkeypatch
    ):
        import asyncio

        app, client, storage = app_test
        sha = _import_blob(client, storage)
        blob = storage / "blobs" / "sha256" / sha[:2] / sha
        q: asyncio.Queue = asyncio.Queue()
        app.state.manager.event_bus.subscribe(q)

        def boom(*a, **k):
            raise OSError(28, "No space left on device")

        # The WRITE half of prune_digest -> set_description -> this.
        monkeypatch.setattr(model_meta_mod, "_write_model_meta_atomic", boom)

        r = client.request("DELETE", "/api/delete", json={"sha256": sha})

        assert r.status_code == 200, (
            "cleanup after an irreversible act must never invert the reported status"
        )
        assert not blob.exists()
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert any(e.get("event") == "blob_changed" for e in events), (
            "blob_changed must still fire -- otherwise the blob is gone and no "
            "other tab is ever told"
        )


# ===================================================================
# The four INFERENCE routes -- read_manifest raises pydantic's
# ValidationError, which is a SIBLING of ManifestValidationError under
# ValueError, NOT a subclass (verified empirically below, not assumed from
# the name): `except ManifestValidationError` never catches it, and with
# zero app-level exception handlers a schema-invalid manifest crashes the
# whole request as a bare, unhandled 500.
#
# ⛔ THESE FOUR ARE PER-TAG ROUTES, LIKE /v1/models/{tag} ABOVE, NOT LISTING
# ROUTES LIKE /v1/models. Both today's crash AND the fix answer 500 -- a
# bare `status_code == 500` assertion is therefore GREEN ON BOTH ARMS and
# proves nothing. The watchable red is the BODY, measured empirically on
# unmodified code before any of these assertions were written:
#     BEFORE (uncaught):  500, plain text "Internal Server Error"
#                          (content-type text/plain; r.json() raises
#                          JSONDecodeError)
#     AFTER  (handled):   500, JSON {"detail": "manifest for '<tag>' is
#                          present but unreadable"} -- get_model's own
#                          precedent (in models.py), reused verbatim.
# ===================================================================

def test_validation_error_is_a_sibling_not_a_subclass_of_manifest_validation_error():
    """The rule this whole fix turns on -- verified against the live classes,
    not taken on trust."""
    assert issubclass(ValidationError, ValueError)
    assert issubclass(ManifestValidationError, ValueError)
    assert not issubclass(ValidationError, ManifestValidationError)


BOGUS_MODEL_MANIFEST = (
    "model_tag: {tag}\ndisplay_name: Bad\n"
    "gguf_blob_sha256: " + SAMPLE_SHA + "\n"
    "gguf_size_bytes: 10000000000\ncontext_size: 4096\n"
    "expected_vram_bytes: 11000000000\nllama_server_flags: {{ctx_size: 4096}}\n"
    "totally_bogus_field_xyz: 1\n"
)
BOGUS_PLUGIN_MANIFEST = (
    "model_tag: {tag}\nkind: plugin\nlane: cpu\n"
    "resource_key: whisperx-main\ncapabilities: [transcribe]\n"
    "totally_bogus_field_xyz: 1\n"
)
# A YAML-syntax leak probe (unclosed flow sequence). Measured: raises
# yaml.parser.ParserError -- a yaml.YAMLError -- NOT pydantic.ValidationError;
# "[unclosed" is invalid YAML and never reaches parse_manifest at all. (The
# pydantic class is NOT what this probe raises -- a comment claiming otherwise
# would be exactly the "a comment that overstates its guard" failure mode
# this fix is itself about.) Same shape as this file's existing YAML_TYPO /
# test_openai_discovery_500_leaks_no_manifest_content precedent. The
# pydantic-ValidationError no-leak property IS covered, just not here: the
# EXACT-equality assertion in test_*_schema_invalid_handled against the
# fixed message would itself break if str(e) ever leaked in, since a
# pydantic ValidationError echoes the offending field name.
SECRET_LEAK_PROBE = (
    "model_tag: {tag}\ndisplay_name: Bad\n"
    "gguf_blob_sha256: " + SAMPLE_SHA + "\n"
    "gguf_size_bytes: 10000000000\ncontext_size: 4096\n"
    "expected_vram_bytes: 11000000000\nllama_server_flags: {{ctx_size: 4096}}\n"
    "secret_flag: [unclosed\n"
)


def _plant_model(app, tag, body=BOGUS_MODEL_MANIFEST):
    (_manifests_dir(app) / f"{tag}.yaml").write_text(body.format(tag=tag))


def _plant_plugin(app, tag, body=BOGUS_PLUGIN_MANIFEST):
    (_manifests_dir(app) / f"{tag}.yaml").write_text(body.format(tag=tag))


_DELIBERATE_MSG = "manifest for '{tag}' is present but unreadable"


class TestFourInferenceRoutesAnswerDeliberately:
    # --- core fix + negative arm: a schema-invalid manifest (extra=
    # "forbid" field) must still be REJECTED, not loaded and not served,
    # at every one of the four sites. ---

    def test_openai_chat_completions_schema_invalid_handled(self, app_test):
        app, client, _ = app_test
        _plant_model(app, "bad-chat")

        r = client.post("/v1/chat/completions",
                         json={"model": "bad-chat", "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-chat")

    def test_ollama_chat_schema_invalid_handled(self, app_test):
        app, client, _ = app_test
        _plant_model(app, "bad-ollama")

        r = client.post("/api/chat",
                         json={"model": "bad-ollama", "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-ollama")

    def test_embeddings_schema_invalid_handled(self, app_test):
        app, client, _ = app_test
        _plant_model(app, "bad-embed")

        r = client.post("/v1/embeddings", json={"model": "bad-embed", "input": "hi"})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-embed")

    def test_plugin_invoke_schema_invalid_handled(self, app_test):
        app, client, _ = app_test
        _plant_plugin(app, "bad-plugin")

        r = client.post("/api/plugins/bad-plugin/invoke", json={"path": "/x", "payload": {}})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-plugin")

    # --- wiring proof: a SECOND, unrelated exception class (UnicodeDecodeError)
    # must also be caught at the actual call site, not just present in the
    # module's tuple -- a module could define a perfect tuple and forget to
    # use it here. ---

    def test_openai_chat_completions_undecodable_handled(self, app_test):
        app, client, _ = app_test
        (_manifests_dir(app) / "bad-chat2.yaml").write_bytes(UNDECODABLE)

        r = client.post("/v1/chat/completions",
                         json={"model": "bad-chat2", "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-chat2")

    def test_ollama_chat_undecodable_handled(self, app_test):
        app, client, _ = app_test
        (_manifests_dir(app) / "bad-ollama2.yaml").write_bytes(UNDECODABLE)

        r = client.post("/api/chat",
                         json={"model": "bad-ollama2", "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-ollama2")

    def test_embeddings_undecodable_handled(self, app_test):
        app, client, _ = app_test
        (_manifests_dir(app) / "bad-embed2.yaml").write_bytes(UNDECODABLE)

        r = client.post("/v1/embeddings", json={"model": "bad-embed2", "input": "hi"})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-embed2")

    def test_plugin_invoke_undecodable_handled(self, app_test):
        app, client, _ = app_test
        (_manifests_dir(app) / "bad-plugin2.yaml").write_bytes(UNDECODABLE)

        r = client.post("/api/plugins/bad-plugin2/invoke", json={"path": "/x", "payload": {}})

        assert r.status_code == 500
        assert r.json()["detail"] == _DELIBERATE_MSG.format(tag="bad-plugin2")

    # --- no-leak: matters MORE here than at discovery (models.py's
    # rule) because these routes take an attacker-controlled request body
    # too, not just a URL tag. ---

    def test_openai_chat_completions_leaks_no_manifest_content(self, app_test):
        app, client, _ = app_test
        _plant_model(app, "leak-chat", body=SECRET_LEAK_PROBE)

        detail = client.post(
            "/v1/chat/completions",
            json={"model": "leak-chat", "messages": [{"role": "user", "content": "hi"}]},
        ).json()["detail"]

        assert "secret_flag" not in detail
        assert "unclosed" not in detail
        assert "line" not in detail.lower()

    def test_ollama_chat_leaks_no_manifest_content(self, app_test):
        app, client, _ = app_test
        _plant_model(app, "leak-ollama", body=SECRET_LEAK_PROBE)

        detail = client.post(
            "/api/chat",
            json={"model": "leak-ollama", "messages": [{"role": "user", "content": "hi"}]},
        ).json()["detail"]

        assert "secret_flag" not in detail
        assert "unclosed" not in detail
        assert "line" not in detail.lower()

    def test_embeddings_leaks_no_manifest_content(self, app_test):
        app, client, _ = app_test
        _plant_model(app, "leak-embed", body=SECRET_LEAK_PROBE)

        detail = client.post(
            "/v1/embeddings", json={"model": "leak-embed", "input": "hi"},
        ).json()["detail"]

        assert "secret_flag" not in detail
        assert "unclosed" not in detail
        assert "line" not in detail.lower()

    def test_plugin_invoke_leaks_no_manifest_content(self, app_test):
        app, client, _ = app_test
        (_manifests_dir(app) / "leak-plugin.yaml").write_text(
            "model_tag: leak-plugin\nkind: plugin\nlane: cpu\n"
            "resource_key: whisperx-main\nsecret_flag: [unclosed\n"
        )

        detail = client.post(
            "/api/plugins/leak-plugin/invoke", json={"path": "/x", "payload": {}},
        ).json()["detail"]

        assert "secret_flag" not in detail
        assert "unclosed" not in detail
        assert "line" not in detail.lower()

    # --- regression guard: FileNotFoundError IS an OSError, so widening
    # these four to catch OSError could easily have swallowed "absent" into
    # "unreadable" too, same trap the drift guard already pins for the
    # discovery/management routes above. ⛔ BOTH-ARMS-GREEN -- absent
    # already answers something-not-500 on unfixed code too (a 404 from the
    # existing FileNotFoundError arm, or a validation error from the
    # inbound-request pydantic model; either way it never reaches the bug).
    # Disclosed and mutation-checked for each of the four sites (OpenAI,
    # Ollama, embeddings, plugin).

    def test_openai_chat_completions_absent_still_404(self, app_test):
        app, client, _ = app_test

        r = client.post("/v1/chat/completions",
                         json={"model": "never-existed", "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 404
        assert r.json()["detail"] == "model not found: never-existed"

    def test_ollama_chat_absent_still_404(self, app_test):
        app, client, _ = app_test

        r = client.post("/api/chat",
                         json={"model": "never-existed", "messages": [{"role": "user", "content": "hi"}]})

        assert r.status_code == 404
        assert r.json()["detail"] == "model not found: never-existed"

    def test_embeddings_absent_still_404(self, app_test):
        app, client, _ = app_test

        r = client.post("/v1/embeddings", json={"model": "never-existed", "input": "hi"})

        assert r.status_code == 404
        assert r.json()["detail"] == "model 'never-existed' has no manifest"

    def test_plugin_invoke_absent_still_404(self, app_test):
        app, client, _ = app_test

        r = client.post("/api/plugins/never-existed/invoke", json={"path": "/x", "payload": {}})

        assert r.status_code == 404
        assert r.json()["detail"] == "plugin not found: never-existed"
