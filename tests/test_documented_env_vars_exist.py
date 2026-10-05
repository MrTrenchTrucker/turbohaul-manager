"""Every TURBOHAUL_* env var named in a document must exist in the env map.

Why this test exists
--------------------
`docs/AI_AGENT_SETUP.md` shipped a copy-pasteable `docker run` line setting
`TURBOHAUL_IDLE_HOT_SECONDS`. The real name is `TURBOHAUL_IDLE_HOT_S`. The
documented name is read nowhere, so the container starts, nothing errors, and the
setting is silently ignored.

That is the failure mode worth guarding: a wrong value that CRASHES gets fixed in
minutes; a wrong value that is IGNORED survives for months. This exact `_SECONDS`
vs `_S` class had already been found and fixed once in an earlier documentation
pass and came back, because nothing pinned the docs to the code.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENV_TOKEN = re.compile(r"\bTURBOHAUL_[A-Z0-9_]+\b")


def _real_env_names() -> set[str]:
    """Every TURBOHAUL_* name the code actually reads, from anywhere in src/.

    Derived, never hand-listed: a hand-kept copy is the same defect this test
    guards, one level up.

    Scanning only config.py's env map was the first attempt and it was WRONG --
    it flagged `TURBOHAUL_CONFIG_PATH`, `TURBOHAUL_ALLOW_PUBLIC_BIND` and five
    others that are read perfectly legitimately in other modules. A gate that
    fails on correct documentation gets switched off, so it must scan the whole
    package.
    """
    names: set[str] = set()
    for py in (REPO / "src").rglob("*.py"):
        names |= set(re.findall(r'"(TURBOHAUL_[A-Z0-9_]+)"', py.read_text(errors="replace")))
        names |= set(re.findall(r"'(TURBOHAUL_[A-Z0-9_]+)'", py.read_text(errors="replace")))
    return names



REMOVAL_WORDS = ("no effect", "removed", "replaced", "deprecat", "no longer",
                 "ignored", "does nothing", "not read")


def _documented_as_removed(text: str, name: str) -> bool:
    """True when the document says, near the mention, that the name is dead.

    Derived from the prose rather than a hand-kept exemption list -- a hand-kept
    list is the same defect this file guards, one level up.
    """
    for m in re.finditer(re.escape(name), text):
        window = text[max(0, m.start() - 300):m.end() + 300].lower()
        if any(w in window for w in REMOVAL_WORDS):
            return True
    return False


def _documented_env_names() -> dict[str, set[str]]:
    """Every TURBOHAUL_* token appearing in any shipped markdown, by file."""
    out: dict[str, set[str]] = {}
    for md in list(REPO.glob("*.md")) + list((REPO / "docs").glob("*.md")):
        names = set(ENV_TOKEN.findall(md.read_text(encoding="utf-8", errors="replace")))
        if names:
            out[str(md.relative_to(REPO))] = names
    return out


def test_config_env_map_is_not_empty():
    """Guard the guard: an empty real-name set would make every check vacuous."""
    real = _real_env_names()
    assert len(real) >= 5, (
        f"only {len(real)} TURBOHAUL_* names found across src/ -- the extractor is "
        "broken, so every assertion below would pass trivially"
    )


def test_documented_corpus_is_not_empty():
    """Guard the guard: if no doc mentions an env var, this test proves nothing."""
    documented = _documented_env_names()
    assert documented, "no TURBOHAUL_* env var found in any document -- nothing was checked"


def test_every_documented_env_var_is_read_by_the_code():
    real = _real_env_names()
    documented = _documented_env_names()
    unknown: list[str] = []
    for doc, names in sorted(documented.items()):
        text = (REPO / doc).read_text(encoding="utf-8", errors="replace")
        for name in sorted(names - real):
            if _documented_as_removed(text, name):
                # A document that explicitly tells the reader this variable no
                # longer does anything is CORRECT, not stale. Failing on it
                # would punish exactly the writing we want -- and a gate that
                # fires on good documentation gets switched off.
                continue
            close = [r for r in real if r.startswith(name.rsplit("_", 1)[0])]
            hint = f"  (did you mean {', '.join(sorted(close))}?)" if close else ""
            unknown.append(f"{doc}: {name}{hint}")
    assert not unknown, (
        "these env vars are documented but read nowhere, so setting them does "
        "nothing and fails silently:\n  " + "\n  ".join(unknown)
    )
