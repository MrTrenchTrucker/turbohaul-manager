"""Static regression tests for the SM86 candidate frontend type contract.

Background: Turbohaul v0.6 commit bc8b8f8 introduced ``engine_op`` in a
synthesized ``ResidentModel`` object (Dashboard.tsx ``synthesizeResident``)
but omitted the property from the ``ResidentModel`` interface in
``src/frontend/src/api.ts``. This caused ``tsc --noEmit`` to fail with
TS2353 ('engine_op does not exist in type ResidentModel'), which blocked
the Dockerfile.sm86 Stage 1 frontend build.

These tests pin the contract statically so a future revert is caught by the
cheap gate (no browser, no Docker, no GPU).
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text()


class TestResidentModelEngineOpContract:
    """ResidentModel must declare engine_op as optional."""

    def test_api_ts_declares_engine_op_on_resident_model(self):
        src = _read("src/frontend/src/api.ts")
        # Locate the ResidentModel interface block
        m = re.search(r"export interface ResidentModel \{(.*?)\}", src, re.S)
        assert m, "ResidentModel interface not found in api.ts"
        block = m.group(1)
        assert "engine_op" in block, (
            "ResidentModel must declare engine_op (omitted since v0.6 bc8b8f8; "
            "synthesizeResident in Dashboard.tsx assigns it)"
        )
        # Must be optional (engine_op?:)
        assert re.search(r"engine_op\s*\?\s*:\s*string", block), (
            "engine_op must be optional (engine_op?: string) -- the backend "
            "residents[] array does not yet carry it; only ActiveInfo/LoadingInfo do"
        )

    def test_dashboard_synthesize_resident_assigns_engine_op(self):
        src = _read("src/frontend/src/components/Dashboard.tsx")
        # synthesizeResident must assign engine_op without an `as any` cast
        assert "engine_op" in src, "synthesizeResident no longer references engine_op"
        # The old cast (source as any).engine_op must be gone
        assert "(source as any).engine_op" not in src, (
            "synthesizeResident still casts to any for engine_op -- the property "
            "is now declared on ResidentModel, the cast is unnecessary and hides regressions"
        )

    def test_dashboard_model_card_reads_engine_op_typed(self):
        src = _read("src/frontend/src/components/Dashboard.tsx")
        # The model card must read model.engine_op without an `as any` cast
        assert "(model as any).engine_op" not in src, (
            "ModelCard still casts model to any for engine_op -- the property "
            "is now declared on ResidentModel"
        )
        assert "model.engine_op" in src, "ModelCard must read model.engine_op"


class TestSm86DockerfileUiOffContract:
    """Dockerfile.sm86 must fully disable UI provisioning, not just BUILD_UI.

    LLAMA_BUILD_UI=OFF alone is insufficient because LLAMA_USE_PREBUILT_UI
    defaults to ON, causing ui-assets.cmake to attempt HF Bucket download
    in the Docker build (network-dependent, fails offline). Both flags must
    be OFF to make the vendored CMake target truly honour UI-off.
    """

    def test_dockerfile_sm86_disables_build_ui(self):
        df = _read("Dockerfile.sm86")
        assert "-DLLAMA_BUILD_UI=OFF" in df, "Dockerfile.sm86 must pass -DLLAMA_BUILD_UI=OFF"

    def test_dockerfile_sm86_disables_prebuilt_ui(self):
        df = _read("Dockerfile.sm86")
        assert "-DLLAMA_USE_PREBUILT_UI=OFF" in df, (
            "Dockerfile.sm86 must pass -DLLAMA_USE_PREBUILT_UI=OFF -- without it "
            "ui-assets.cmake attempts HF Bucket download (LLAMA_USE_PREBUILT_UI "
            "defaults to ON), failing the Docker build offline"
        )

    def test_dockerfile_sm86_no_html_stubs(self):
        df = _read("Dockerfile.sm86")
        # Must not create any fake/empty HTML stubs for the UI
        assert "index.html" not in df.lower() or "stub" not in df.lower(), (
            "Dockerfile.sm86 must not contain HTML stub workarounds"
        )
        assert r"echo.*<!DOCTYPE\|echo.*<html" not in df.lower(), (
            "Dockerfile.sm86 must not write inline HTML stubs"
        )

    def test_dockerfile_sm86_comments_correct(self):
        df = _read("Dockerfile.sm86")
        # Must not claim six architectures
        assert "6 architectures" not in df, "Dockerfile.sm86 still claims 6 architectures"
        assert "six arch" not in df.lower(), "Dockerfile.sm86 still claims six arches"
        # Must not reference the old Dockerfile name
        assert "Dockerfile.cuda-multi" not in df, "Dockerfile.sm86 still references Dockerfile.cuda-multi"
        # Must not reference the old version tag
        assert "v0.6.0" not in df, "Dockerfile.sm86 still references v0.6.0"
        # Must not list sm_75/sm_89/sm_90/sm_120 (multi-arch leftovers)
        for arch in ["sm_75", "sm_89", "sm_90", "sm_120"]:
            assert arch not in df, f"Dockerfile.sm86 still references {arch}"
