"""Regression tests for the blob-delete reference guard: DELETE /api/delete must refuse to
unlink a blob that any manifest still references, in ANY of its three blob
fields -- and the manifest summary rows must expose all three so the FE can
count references correctly in the first place.

The hazard: a blob can be referenced ONLY as a projector or a
speculative draft model, never as a manifest's own gguf. The Models page
groups tiles on gguf_blob_sha256 alone, so it counted zero manifests for
such a blob, rendered it as an empty model reading "No manifests yet",
and offered Delete. The API obeyed. Re-acquiring a drafter that many
manifests speculate against, or a projector that carries vision for
several models, is a multi-GB download,
not a rollback.

Two traps this file exists to pin:

1. THE EMPTY STRING. mmproj_blob_sha256 and spec_draft_gguf_blob_sha256 are
   the EMPTY STRING on most manifests, not null (manifest.py
   explicitly allows ""). A comparison without a truthiness guard, or a
   containment/prefix test against a short digest, matches everything and
   409s every delete -- dead in a way that looks like it is working.

2. THE UNREADABLE MANIFEST. gguf_blob_sha256 requires 64 hex with NO empty
   escape (see manifest.py), unlike its two siblings. A manifest carrying
   `gguf_blob_sha256: ''` therefore fails pydantic validation outright, so a
   guard built on read_manifest() skips it -- and skipping a row in the guard
   whose whole job is preventing data loss is a hole in the guard. This is
   the unreadable-row family, and it is not hypothetical: a realistic fixture
   has exactly that shape.
"""
import pytest
from fastapi.testclient import TestClient

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
from turbohaul.manifest import ModelManifest, PluginManifest, write_manifest_atomic


OTHER_SHA = "a" * 64
ABSENT_SHA = "b" * 64


@pytest.fixture
def app_test(tmp_path):
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
            llama_server_binary=tmp_path / "fake",
            default_port_base=59500,
        ),
        ui=UIConfig(static_path=tmp_path / "ui"),
    )
    runtime = RuntimeConfig(queue=QueueConfig(safety_enabled=False), pull=PullConfig())
    app = create_app(boot, runtime, auto_start_worker=False, auto_boot_reconcile=False)
    with TestClient(app) as client:
        yield app, client, storage_root


def _import_blob(client, storage_root, body: bytes = b"payload") -> str:
    """Put a real blob in the store via the import route. Returns its sha256."""
    src = storage_root / "import-staging" / f"m-{body.hex()}.gguf"
    src.write_bytes(GGUF_MAGIC + body)
    r = client.post("/api/import", json={"path": str(src)})
    assert r.status_code == 200, r.text
    return r.json()["sha256"]


def _blob_file(storage_root, sha: str):
    return storage_root / "blobs" / "sha256" / sha[:2] / sha


def _manifests_dir(app):
    return app.state.manager.boot.storage.manifests_path


def _mk_model(**overrides) -> ModelManifest:
    base = dict(
        model_tag="a-model",
        display_name="A Model",
        gguf_blob_sha256=OTHER_SHA,
        gguf_size_bytes=22_000_000_000,
        context_size=4096,
        expected_vram_bytes=22_500_000_000,
        llama_server_flags={"ctx_size": 4096},
    )
    base.update(overrides)
    return ModelManifest(**base)


def _mk_plugin(**overrides) -> PluginManifest:
    base = dict(
        model_tag="whisperx-transcribe",
        kind="plugin",
        lane="cpu",
        resource_key="whisperx-main",
        capabilities=["transcribe"],
    )
    base.update(overrides)
    return PluginManifest(**base)


def _write_raw(app, tag: str, text: str) -> None:
    """Write a manifest file verbatim, bypassing validation."""
    (_manifests_dir(app) / f"{tag}.yaml").write_text(text)


def _delete_blob(client, sha: str):
    return client.request("DELETE", "/api/delete", json={"sha256": sha})


class TestReferencedBlobIsRefused:
    """A blob named by any manifest, in any of the three fields, must survive."""

    def test_409_when_referenced_as_mmproj_only(self, app_test):
        """THE CORE CASE: a blob referenced only as a projector -- no tile counts it."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="vision-model", mmproj_blob_sha256=sha)
        )

        r = _delete_blob(client, sha)

        assert r.status_code == 409, f"expected 409, got {r.status_code}: {r.text[:300]}"
        detail = r.json()["detail"]
        assert "vision-model" in detail, detail
        assert "mmproj_blob_sha256" in detail, detail
        # Asserting the status without asserting the file is exactly the hole
        # that makes a guard look fixed while it still unlinks.
        assert _blob_file(storage, sha).exists(), "409 returned but the blob was UNLINKED"

    def test_409_when_referenced_as_spec_draft_only(self, app_test):
        """A blob referenced only as a draft model -- a drafter that many manifests speculate against."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        write_manifest_atomic(
            _manifests_dir(app),
            _mk_model(model_tag="drafting-model", spec_draft_gguf_blob_sha256=sha),
        )

        r = _delete_blob(client, sha)

        assert r.status_code == 409, r.text[:300]
        detail = r.json()["detail"]
        assert "drafting-model" in detail, detail
        assert "spec_draft_gguf_blob_sha256" in detail, detail
        assert _blob_file(storage, sha).exists()

    def test_409_when_referenced_as_gguf(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="main-model", gguf_blob_sha256=sha)
        )

        r = _delete_blob(client, sha)

        assert r.status_code == 409, r.text[:300]
        assert "main-model" in r.json()["detail"]
        assert _blob_file(storage, sha).exists()

    def test_409_names_every_referencing_manifest(self, app_test):
        """A shared projector takes vision off THREE models; the operator needs all
        three names, not just the first one found."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        for tag in ("qwen-a", "qwen-b", "qwen-c"):
            write_manifest_atomic(
                _manifests_dir(app), _mk_model(model_tag=tag, mmproj_blob_sha256=sha)
            )

        r = _delete_blob(client, sha)

        assert r.status_code == 409
        detail = r.json()["detail"]
        for tag in ("qwen-a", "qwen-b", "qwen-c"):
            assert tag in detail, f"{tag} missing from: {detail}"
        assert "3 manifest(s)" in detail, detail

    def test_a_refused_delete_does_not_prune_the_description(self, app_test):
        """prune_digest() must stay BELOW the guard. A refused delete that
        still dropped the blob's model_meta entry would destroy operator-written
        metadata on the exact blob it just declined to destroy."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        assert client.put(
            f"/api/blobs/{sha}/description", json={"description": "keep me"}
        ).status_code == 200
        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="holder", mmproj_blob_sha256=sha)
        )

        assert _delete_blob(client, sha).status_code == 409

        from turbohaul.model_meta import read_model_meta

        assert sha in read_model_meta(_manifests_dir(app)), "description pruned on a REFUSAL"


class TestTheEmptyStringTrap:
    """'' is a substring of every string and a prefix of every string. A guard
    that compares without a truthiness check 409s everything."""

    def test_empty_blob_fields_do_not_block_an_unrelated_blob(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        # The overwhelmingly common live shape: a text-only model with both
        # optional blob fields stored as "", not null.
        m = _mk_model(
            model_tag="text-only", mmproj_blob_sha256="", spec_draft_gguf_blob_sha256=""
        )
        assert m.mmproj_blob_sha256 == ""
        assert m.spec_draft_gguf_blob_sha256 == ""
        write_manifest_atomic(_manifests_dir(app), m)

        r = _delete_blob(client, sha)

        assert r.status_code == 200, (
            f"the '' trap: an UNREFERENCED blob was refused -- {r.status_code} {r.text[:300]}"
        )
        assert not _blob_file(storage, sha).exists()

    def test_helper_refuses_to_scan_for_an_empty_digest(self, app_test):
        """Pins the trap at the boundary it actually lives on. Through the route
        an empty sha is already rejected at import_.py's `if not sha` 400, so
        neither truthiness guard can be reached -- but _manifests_referencing()
        is where the damage would be done, and '' == '' is a real match against
        the empty mmproj field of every text-only manifest on disk.

        Killed only by removing BOTH guards (the is_valid_digest() gate AND the
        per-value truthiness check); either alone is sufficient, which is the
        point of guarding at a boundary rather than trusting a caller.
        """
        app, client, _ = app_test
        from turbohaul.api.import_ import _manifests_referencing

        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="text-only", mmproj_blob_sha256="")
        )

        assert _manifests_referencing(_manifests_dir(app), "") == []

    def test_helper_refuses_to_scan_for_a_short_digest(self, app_test):
        """A short digest is a substring of half the registry through the
        raw-text fallback -- the same trap, through the other door."""
        app, client, _ = app_test
        from turbohaul.api.import_ import _manifests_referencing

        _write_raw(app, "corrupt", "{[[ deadbeef :: not yaml ]]}\n\t- \x00")

        assert _manifests_referencing(_manifests_dir(app), "deadbeef") == []

    def test_empty_fields_in_an_unparseable_file_do_not_block_either(self, app_test):
        """Same trap through the raw-text door, where '' matching is worse:
        `sha in ""` is False but `"" in text` is True for every file."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        _write_raw(app, "broken", "mmproj_blob_sha256: ''\n  : : not yaml [\n")

        r = _delete_blob(client, sha)

        assert r.status_code == 200, r.text[:300]
        assert not _blob_file(storage, sha).exists()


class TestUnreadableAndUnparseableManifests:
    def test_a_pydantic_invalid_manifest_still_blocks(self, app_test):
        """A pydantic-invalid manifest's shape. `gguf_blob_sha256: ''` fails
        the digest rule in manifest.py (64 hex, no empty escape) while the file is perfectly
        good YAML, so read_manifest() raises ValidationError. A guard that
        skipped it would return 200 and unlink a projector three models need."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        _write_raw(
            app,
            "zzz-scratch",
            "model_tag: zzz-scratch\n"
            "display_name: scratch\n"
            "gguf_blob_sha256: ''\n"
            f"mmproj_blob_sha256: '{sha}'\n"
            "spec_draft_gguf_blob_sha256: ''\n"
            "gguf_size_bytes: 4096\n"
            "context_size: 512\n"
            "expected_vram_bytes: 1\n"
            "hidden: true\n"
            "revision: 1\n",
        )
        # The premise of the test: this file is genuinely unreadable as a Manifest.
        from pydantic import ValidationError

        from turbohaul.manifest import read_manifest

        with pytest.raises(ValidationError):
            read_manifest(_manifests_dir(app), "zzz-scratch")

        r = _delete_blob(client, sha)

        assert r.status_code == 409, (
            f"a pydantic-invalid manifest's reference was skipped: "
            f"{r.status_code} {r.text[:300]}"
        )
        assert "zzz-scratch" in r.json()["detail"]
        assert _blob_file(storage, sha).exists()

    def test_unparseable_yaml_naming_the_digest_blocks_and_says_so(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        _write_raw(app, "corrupt", f"{{[[ not yaml at all :: {sha} ]]}}\n\t- \x00")

        r = _delete_blob(client, sha)

        assert r.status_code == 409, r.text[:300]
        detail = r.json()["detail"]
        assert "corrupt" in detail, detail
        # The operator must be told it is a BROKEN FILE, not a real reference,
        # or they go hunting a phantom manifest entry.
        assert "unparseable" in detail, detail
        assert _blob_file(storage, sha).exists()

    def test_one_corrupt_file_does_not_brick_every_delete(self, app_test):
        """The reason the fallback is a digest match and not a blanket refusal."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        _write_raw(app, "corrupt", "{[[ not yaml at all :: ]]}\n\t- \x00")

        r = _delete_blob(client, sha)

        assert r.status_code == 200, (
            f"a corrupt manifest that does not name this digest blocked it: {r.text[:300]}"
        )
        assert not _blob_file(storage, sha).exists()

    def test_a_digest_in_a_parsed_manifests_description_does_not_block(self, app_test):
        """Structure is authoritative when it is available: a text scan over a
        readable manifest would refuse on a digest merely quoted in prose."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        write_manifest_atomic(
            _manifests_dir(app),
            _mk_model(model_tag="chatty", description=f"was built from {sha}"),
        )

        r = _delete_blob(client, sha)

        assert r.status_code == 200, r.text[:300]
        assert not _blob_file(storage, sha).exists()


class TestCaseInsensitivity:
    def test_uppercase_digest_in_a_parsed_manifest_still_blocks(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        _write_raw(
            app,
            "shouty",
            "model_tag: shouty\n"
            f"gguf_blob_sha256: '{OTHER_SHA}'\n"
            f"mmproj_blob_sha256: '{sha.upper()}'\n"
            "gguf_size_bytes: 4096\n"
            "context_size: 512\n"
            "expected_vram_bytes: 1\n",
        )

        r = _delete_blob(client, sha)

        assert r.status_code == 409, r.text[:300]
        assert "shouty" in r.json()["detail"]
        assert _blob_file(storage, sha).exists()

    def test_uppercase_digest_in_an_unparseable_file_still_blocks(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        _write_raw(app, "shouty-corrupt", f"{{[[ :: {sha.upper()} ]]}}\n\t- \x00")

        r = _delete_blob(client, sha)

        assert r.status_code == 409, r.text[:300]
        assert _blob_file(storage, sha).exists()


class TestUnreferencedBlobsStillDelete:
    def test_unreferenced_blob_deletes_with_manifests_present(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage, b"target")
        other = _import_blob(client, storage, b"other")
        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="unrelated", gguf_blob_sha256=other)
        )
        write_manifest_atomic(_manifests_dir(app), _mk_plugin())

        r = _delete_blob(client, sha)

        assert r.status_code == 200, r.text[:300]
        assert not _blob_file(storage, sha).exists()

    def test_no_manifests_at_all_still_deletes(self, app_test):
        app, client, storage = app_test
        sha = _import_blob(client, storage)

        assert _delete_blob(client, sha).status_code == 200
        assert not _blob_file(storage, sha).exists()

    def test_the_guard_is_not_sticky(self, app_test):
        """The whole reason the 409 is absolute with no force flag: the guard
        stores nothing and re-reads the directory every request, so removing
        the last reference makes the SAME delete succeed. If this test ever
        goes red, an override genuinely becomes necessary."""
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="holder", mmproj_blob_sha256=sha)
        )

        assert _delete_blob(client, sha).status_code == 409
        assert client.delete("/api/manifests/holder").status_code == 200

        r = _delete_blob(client, sha)

        assert r.status_code == 200, (
            f"the orphan could not be removed after its last reference went away: "
            f"{r.status_code} {r.text[:300]}"
        )
        assert not _blob_file(storage, sha).exists()


class TestExistingStatusCodesUnchanged:
    def test_404_unknown_digest(self, app_test):
        app, client, _ = app_test
        assert _delete_blob(client, "f" * 64).status_code == 404

    def test_400_malformed_digest(self, app_test):
        app, client, _ = app_test
        assert _delete_blob(client, "not-a-sha256").status_code == 400

    def test_400_missing_sha(self, app_test):
        app, client, _ = app_test
        assert client.request("DELETE", "/api/delete", json={}).status_code == 400

    def test_absent_but_referenced_digest_is_404_not_409(self, app_test):
        """Existence is checked FIRST. Deleting a file that is
        not there is a no-op with nothing to protect, and telling the caller to
        remove references to a file that does not exist is a worse instruction
        than "not found". The broken state is real and worth surfacing -- in
        the listing or a health check, not as a delete refusal."""
        app, client, _ = app_test
        write_manifest_atomic(
            _manifests_dir(app), _mk_model(model_tag="dangling", mmproj_blob_sha256=ABSENT_SHA)
        )

        r = _delete_blob(client, ABSENT_SHA)

        assert r.status_code == 404, (
            f"a digest that is referenced but ABSENT must keep its 404: "
            f"{r.status_code} {r.text[:300]}"
        )


class TestSummaryRowsCarryAllThreeBlobFields:
    """Without these two keys the FE cannot count a projector-only or
    drafter-only reference, which is what made the tile read "No manifests
    yet" and offer Delete in the first place."""

    def test_model_row_carries_both_new_fields(self, app_test):
        app, client, _ = app_test
        write_manifest_atomic(
            _manifests_dir(app),
            _mk_model(
                model_tag="full-model",
                mmproj_blob_sha256="c" * 64,
                spec_draft_gguf_blob_sha256="d" * 64,
            ),
        )

        row = {r["model_tag"]: r for r in client.get("/api/manifests").json()["manifests"]}[
            "full-model"
        ]

        assert row["gguf_blob_sha256"] == OTHER_SHA
        assert row["mmproj_blob_sha256"] == "c" * 64
        assert row["spec_draft_gguf_blob_sha256"] == "d" * 64

    def test_empty_optional_fields_serialise_as_empty_string_not_null(self, app_test):
        """The FE must be able to tell "" (no projector) from None (a row that
        could not be read). Collapsing them would reintroduce the original hole as a
        false-positive instead of a false-negative."""
        app, client, _ = app_test
        write_manifest_atomic(_manifests_dir(app), _mk_model(model_tag="text-only"))

        row = {r["model_tag"]: r for r in client.get("/api/manifests").json()["manifests"]}[
            "text-only"
        ]

        assert row["mmproj_blob_sha256"] == ""
        assert row["spec_draft_gguf_blob_sha256"] == ""

    def test_plugin_row_gets_none_for_all_three(self, app_test):
        """PluginManifest declares none of the three and this listing is
        deliberately not kind-filtered, so a direct attribute read would
        AttributeError on every plugin row."""
        app, client, _ = app_test
        write_manifest_atomic(_manifests_dir(app), _mk_plugin())

        row = {r["model_tag"]: r for r in client.get("/api/manifests").json()["manifests"]}[
            "whisperx-transcribe"
        ]

        assert row["kind"] == "plugin"
        assert row["gguf_blob_sha256"] is None
        assert row["mmproj_blob_sha256"] is None
        assert row["spec_draft_gguf_blob_sha256"] is None

    def test_unreadable_row_carries_both_new_fields_as_none(self, app_test):
        app, client, _ = app_test
        _write_raw(app, "bad-model", "model_tag: bad-model\ngguf_blob_sha256: ''\n")

        row = {r["model_tag"]: r for r in client.get("/api/manifests").json()["manifests"]}[
            "bad-model"
        ]

        assert row["error"] == "unreadable"
        assert row["mmproj_blob_sha256"] is None
        assert row["spec_draft_gguf_blob_sha256"] is None

    def test_the_two_row_shapes_stay_uniform(self, app_test):
        """Uniform-row rule: a consumer must not hit a missing key on exactly the
        row that is already broken. Pins the ONE pre-existing asymmetry
        (etag/error) so any NEW divergence fails here rather than in a browser."""
        app, client, _ = app_test
        write_manifest_atomic(_manifests_dir(app), _mk_model(model_tag="good-model"))
        _write_raw(app, "bad-model", "model_tag: bad-model\ngguf_blob_sha256: ''\n")

        rows = {r["model_tag"]: r for r in client.get("/api/manifests").json()["manifests"]}
        good, bad = set(rows["good-model"]), set(rows["bad-model"])

        assert good - bad == {"etag"}, f"new key missing from the unreadable row: {good - bad}"
        assert bad - good == {"error"}, f"unexpected extra key on the unreadable row: {bad - good}"


class TestPerTagGetUnchanged:
    def test_get_manifest_by_tag_adds_no_new_keys(self, app_test):
        """The per-tag route is unaffected: it adds no new keys."""
        app, client, _ = app_test
        m = _mk_model(model_tag="untouched", mmproj_blob_sha256="c" * 64)
        write_manifest_atomic(_manifests_dir(app), m)

        r = client.get("/api/manifests/untouched")

        assert r.status_code == 200
        assert r.headers["ETag"] == '"1"'
        # model_dump() keys + the one derived marker already added.
        assert set(r.json()) == set(m.model_dump(mode="json")) | {"cache_reuse_inert_mmproj"}


# --- Additional case: the fixture shape the cases above could not produce ---

def _write_bytes(app, tag: str, raw: bytes) -> None:
    """Write a manifest file as RAW BYTES.

    ⛔ `_write_raw` CANNOT be used for these fixtures. It calls `write_text`,
    which always encodes UTF-8, so everything it produces is decodable BY
    CONSTRUCTION -- which is exactly why
    `test_one_corrupt_file_does_not_brick_every_delete` above could never fire
    the shape it is named for. A non-UTF-8 manifest has to be written as bytes.
    """
    (_manifests_dir(app) / f"{tag}.yaml").write_bytes(raw)


# 0xFF/0xFE/0xFD are not legal UTF-8 lead bytes, so `path.read_text()` raises
# UnicodeDecodeError on this -- a ValueError, NOT an OSError, which is how it
# escaped both handlers in _manifests_referencing and 500'd the route.
_NON_UTF8_MANIFEST = b"model_tag: broken\ngguf_blob_sha256: \xff\xfe\xfd\xff\n"


class TestNonUtf8ManifestDoesNotBrickEveryDelete:
    """Additional case. `UnicodeDecodeError` is a ValueError, not an OSError, so
    it escaped `except FileNotFoundError` and `except OSError` alike, propagated
    out of the helper and out of `delete_blob_route` (which has no try/except
    around that call), and FastAPI 500'd. Because the scan walks the whole
    manifests directory, ONE undecodable file killed delete for EVERY blob.

    Before/after on the same blob: no corrupt file -> 200;
    plant one non-UTF-8 .yaml -> 500; remove it -> 200.

    Design choice: an undecodable file joins the UNREADABLE arm, not the
    text fallback. The function's own comment says the fallback exists so one
    corrupt file cannot "either brick every delete or silently open a hole" --
    decoding with errors="replace" would be that second thing, because the
    fallback's premise ("64 hex characters do not occur by accident") only
    holds while every byte decoded intact. If the corruption lands ON the
    digest, the needle misses and a referenced blob is unlinked, silently.
    """

    def test_non_utf8_manifest_refuses_with_a_named_409_not_a_500(self, app_test):
        """AVAILABILITY: the service answers deliberately instead of throwing.

        NOT a 200: under this design an undecodable file appends
        "(unreadable file)" unconditionally, exactly as an unreadable one does,
        so even an UNREFERENCED blob is refused. The win is a deliberate,
        actionable 409 that names the file -- not an unhandled 500.
        """
        app, client, storage = app_test
        sha = _import_blob(client, storage)          # nothing references it
        _write_bytes(app, "badbytes", _NON_UTF8_MANIFEST)

        r = _delete_blob(client, sha)

        assert r.status_code != 500, (
            "UnicodeDecodeError escaped _manifests_referencing and 500'd the "
            "route -- one undecodable manifest bricks EVERY blob delete in the "
            f"system: {r.text[:300]}")
        assert r.status_code == 409, (
            f"expected a deliberate 409, got {r.status_code}: {r.text[:300]}")
        detail = r.json()["detail"]
        assert "badbytes" in detail, (
            "the refusal must NAME the offending file so an operator can clear "
            f"it -- an unnamed refusal is not actionable: {detail}")
        assert _blob_file(storage, sha).exists()

    def test_a_referenced_blob_is_still_refused_when_a_non_utf8_file_is_present(
            self, app_test):
        """FAIL-CLOSED PRESERVED -- the one that matters more.

        Fixing the crash must not re-open the original hole. A blob that IS
        genuinely referenced must still be refused with the corrupt file
        present.
        """
        app, client, storage = app_test
        sha = _import_blob(client, storage)
        write_manifest_atomic(
            _manifests_dir(app),
            _mk_model(model_tag="vision-model", mmproj_blob_sha256=sha),
        )
        _write_bytes(app, "badbytes", _NON_UTF8_MANIFEST)

        r = _delete_blob(client, sha)

        assert r.status_code == 409, f"expected 409, got {r.status_code}: {r.text[:300]}"
        detail = r.json()["detail"]
        # POSITIVE CONTROL, and it is load-bearing: under this design the
        # undecodable file refuses ON ITS OWN, so a bare 409 assertion here
        # would pass even if the reference scan were completely broken. Assert
        # the REAL reference is still found and named.
        assert "vision-model" in detail and "mmproj_blob_sha256" in detail, (
            "the genuine reference is no longer reported -- the crash would be "
            f"fixed but the fail-closed guarantee lost: {detail}")
        assert "badbytes" in detail, detail
        assert _blob_file(storage, sha).exists(), "409 returned but the blob was UNLINKED"
