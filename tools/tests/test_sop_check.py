"""Tests for tools/sop_check.py's structure check. One planted-fault red arm per
category of the structure check (six, listed below), plus the missing-word-budget-key arm, plus one
clean arm proving a valid tree passes outright. Every fixture is built from scratch in a tempdir; nothing
here touches the real repository tree.

The six categories:
  - a missing folder;
  - a missing card or README;
  - a card that disagrees with the registry `public` or the real exports;
  - an outside import past the package entry;
  - a capped file that grew;
  - a stale `MODULE_MAP.md`.

Note: stdlib only (unittest, tempfile, os, shutil) -- no docker, no network, no app.* imports, no third-party
test runner dependency. Runs standalone: `python3 -m unittest tools/tests/test_sop_check.py -v` from the repo
root, or under pytest (pytest collects plain unittest.TestCase classes without this file importing pytest).
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sop_check  # noqa: E402

VALID_ARCHITECTURE = "# Architecture\n\nA tiny fixture repo for the sop_check red-arm proof.\n"

VALID_AGENTS_MD = """# widget

## Purpose
A test fixture module.

## Owns
Test fixture code.

## Does Not Own
Everything else.

## Public Interface
- `foo()`: a trivial function.
- `BAR`: a trivial constant.

## Depends On
- none

## Invariants
- none

## Test Locations
- tools/tests/test_sop_check.py (this file, as a fixture consumer)

## Known Gotchas
- none
"""

VALID_README = "# widget\n\nA test fixture module for the sop_check red-arm proof.\n"

VALID_INIT = '"""Fixture package -- the only entry point (module import contract)."""\nfrom widget.core import foo, BAR\n\n__all__ = ["foo", "BAR"]\n'

VALID_CORE = "def foo():\n    \"\"\"A trivial fixture function.\"\"\"\n    return 1\n\n\nBAR = 1\n"

VALID_MODULES_TOML = """[project]
name = "fixture"
line_cap_soft = 300
line_cap_hard = 500
architecture_word_budget = 1500

[modules.widget]
path = "widget"
owns = "test fixture code"
does_not_own = "everything else"
depends_on = []
public = ["foo", "BAR"]
card = "widget/AGENTS.md"
"""

GADGET_AGENTS_MD = """# gadget

## Purpose
A second test fixture module, used only as an import-contract violation target.

## Owns
Test fixture code.

## Does Not Own
Everything else.

## Public Interface
- `helper()`: a trivial function.

## Depends On
- none

## Invariants
- none

## Test Locations
- tools/tests/test_sop_check.py (this file, as a fixture consumer)

## Known Gotchas
- none
"""

GADGET_README = "# gadget\n\nA second test fixture module for the sop_check red-arm proof.\n"
GADGET_TOML_ENTRY = (
    '\n[modules.gadget]\n'
    'path = "gadget"\n'
    'owns = "test fixture code (import-contract target)"\n'
    'does_not_own = "everything else"\n'
    'depends_on = []\n'
    'public = ["helper"]\n'
    'card = "gadget/AGENTS.md"\n'
)


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


class _FixtureRepo(unittest.TestCase):
    """Base class: every test gets a fresh, valid tempdir repo via self._valid_repo()."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='sop_check_fixture_')
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def _valid_repo(self, modules_toml=VALID_MODULES_TOML):
        root = self._tmp
        _write(os.path.join(root, 'ARCHITECTURE.md'), VALID_ARCHITECTURE)
        _write(os.path.join(root, 'modules.toml'), modules_toml)
        _write(os.path.join(root, 'widget', 'AGENTS.md'), VALID_AGENTS_MD)
        _write(os.path.join(root, 'widget', 'README.md'), VALID_README)
        _write(os.path.join(root, 'widget', '__init__.py'), VALID_INIT)
        _write(os.path.join(root, 'widget', 'core.py'), VALID_CORE)
        return root


class TestCleanArm(_FixtureRepo):
    """A valid tree must pass outright -- proves the checker doesn't cry wolf."""

    def test_valid_tree_passes_with_zero_fails(self):
        root = self._valid_repo()
        sop_check.check(root, write=True)  # generate MODULE_MAP.md + .sop/function_index.json
        fails, warns = sop_check.check(root, write=False)
        self.assertEqual(fails, [], f'a valid fixture tree must have zero failures, got: {fails}')


class TestRedArms(_FixtureRepo):
    """One planted fault per structure-check category (six), plus the missing-budget arm."""

    def test_arm1_a_missing_folder(self):
        """The checker must CATCH a missing registered folder -- a clean,
        named FAIL line, never a crash. A test that accepted any crash would also pass an unrelated one."""
        root = self._valid_repo(VALID_MODULES_TOML.replace('path = "widget"', 'path = "does_not_exist"'))
        sop_check.check(root, write=True)  # must not raise; generates MODULE_MAP.md for the (empty) live set
        fails, _ = sop_check.check(root, write=False)  # must not raise
        self.assertEqual(
            fails, ['widget: registered path does_not_exist does not exist'],
            f'expected exactly one clean FAIL line naming the module and the bad path, got: {fails}',
        )

    def test_arm2_a_missing_card_or_readme(self):
        root = self._valid_repo()
        os.remove(os.path.join(root, 'widget', 'README.md'))
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any('widget/README.md is missing' in f for f in fails),
            f'expected a missing-README failure, got: {fails}',
        )

    def test_arm3_a_card_that_disagrees_with_the_registry_public_or_the_real_exports(self):
        root = self._valid_repo(VALID_MODULES_TOML.replace(
            'public = ["foo", "BAR"]', 'public = ["foo", "BAR", "nonexistent_export"]'
        ))
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("the registry lists `nonexistent_export` as public, but the card's Public Interface does not" in f
                for f in fails),
            f'expected a card/registry disagreement failure, got: {fails}',
        )

    def test_arm4_an_outside_import_past_the_package_entry(self):
        root = self._valid_repo(VALID_MODULES_TOML + GADGET_TOML_ENTRY)
        _write(os.path.join(root, 'gadget', 'AGENTS.md'), GADGET_AGENTS_MD)
        _write(os.path.join(root, 'gadget', 'README.md'), GADGET_README)
        # gadget does NOT reach through widget's entry point (`from widget import foo`); it reaches directly
        # into widget/core.py -- wrong even though gadget doesn't even claim a dependency on widget.
        _write(os.path.join(root, 'gadget', 'helper.py'), "from widget.core import foo\n\n\ndef helper():\n    return foo()\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("reaching past widget's package entry (__init__.py) directly into widget/core.py" in f for f in fails),
            f'expected a package-entry-bypass failure, got: {fails}',
        )

    def test_arm5_a_capped_file_that_grew(self):
        tiny_cap_toml = VALID_MODULES_TOML.replace('line_cap_hard = 500', 'line_cap_hard = 5')
        root = self._valid_repo(tiny_cap_toml)  # VALID_CORE is > 5 lines
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any('widget/core.py' in f and 'hard cap' in f for f in fails),
            f'expected a hard-line-cap failure on widget/core.py, got: {fails}',
        )

    def test_arm6_a_stale_module_map(self):
        root = self._valid_repo()
        sop_check.check(root, write=True)  # generate against the ORIGINAL registry
        # A second, unregistered-yet module folder wouldn't move MODULE_MAP.md (it only reflects the
        # registry) -- so stale it the direct way: change what's IN the registry without regenerating.
        toml = VALID_MODULES_TOML.replace('owns = "test fixture code"', 'owns = "test fixture code, RENAMED"')
        _write(os.path.join(root, 'modules.toml'), toml)
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any('MODULE_MAP.md is stale or missing' in f for f in fails),
            f'expected a stale-MODULE_MAP.md failure, got: {fails}',
        )

    def test_arm7_missing_word_budget_key_is_loud_not_checked(self):
        """A missing architecture_word_budget key must fail LOUDLY, named NOT-CHECKED --
        never a silent skip, and never a KeyError crash."""
        toml_no_budget = VALID_MODULES_TOML.replace('architecture_word_budget = 1500\n', '')
        self.assertNotIn('architecture_word_budget', toml_no_budget, 'fixture setup bug: the key is still present')
        root = self._valid_repo(toml_no_budget)
        fails, _ = sop_check.check(root, write=False)  # must not raise KeyError
        self.assertTrue(
            any(f.startswith('architecture_word_budget: NOT-CHECKED') for f in fails),
            f'expected a loud NOT-CHECKED failure, got: {fails}',
        )


class TestLegacyImportScan(_FixtureRepo):
    """The import-contract check must catch a LEGACY file
    (outside every registered module) reaching past a registered package's entry point -- the common
    shape on a real repository tree, where only a few modules are registered and
    most real importers of them are legacy code. The existing two-registered-modules shape (arm 4 /
    5b above) stays as an extra; it would not fire on such a tree by itself (without
    this scan, planted violations of all four shapes pass at rc 0)."""

    def test_form1_plain_from_import_of_an_internal_file(self):
        root = self._valid_repo()
        _write(os.path.join(root, 'legacy.py'), "from widget.core import foo\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("legacy.py imports widget.core, reaching past widget's package entry" in f for f in fails),
            f'expected a legacy plain-from-import violation, got: {fails}',
        )

    def test_form2_absolute_import_of_an_internal_file(self):
        root = self._valid_repo()
        _write(os.path.join(root, 'legacy.py'), "import widget.core\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("legacy.py imports widget.core, reaching past widget's package entry" in f for f in fails),
            f'expected a legacy absolute-import violation, got: {fails}',
        )

    def test_form3_function_level_from_import_of_an_internal_file(self):
        root = self._valid_repo()
        _write(os.path.join(root, 'legacy.py'), "def f():\n    from widget.core import foo\n    return foo()\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("legacy.py imports widget.core, reaching past widget's package entry" in f for f in fails),
            f'expected a function-level legacy-import violation, got: {fails}',
        )

    def test_form4_from_package_import_submodule_name(self):
        root = self._valid_repo()
        _write(os.path.join(root, 'legacy.py'), "from widget import core\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("legacy.py does `from widget import core`, reaching past widget's package entry" in f for f in fails),
            f'expected a from-package-import-submodule violation, got: {fails}',
        )

    def test_clean_arm_legacy_file_through_the_entry_point_passes(self):
        """A legacy file importing the PUBLIC re-exported name through the entry point (never
        reaching into core.py) must not be flagged -- proves the scan doesn't cry wolf."""
        root = self._valid_repo()
        _write(os.path.join(root, 'legacy.py'), "from widget import foo\n\nfoo()\n")
        sop_check.check(root, write=True)
        fails, _ = sop_check.check(root, write=False)
        self.assertEqual(fails, [], f'a legacy file using the entry point correctly must not be flagged, got: {fails}')

    def test_ancestor_chain_gap_fails_loudly_not_silently(self):
        """A module whose ancestor chain has a GAP (a directory
        missing __init__.py with ANOTHER package directory above it) makes the true import root
        ambiguous from folder structure alone. The checker must say so loudly, never silently
        derive a name (which could make it silently miss every real violation for that module) and
        never silently skip the module out of the scan without saying why."""
        root = self._tmp
        _write(os.path.join(root, 'ARCHITECTURE.md'), VALID_ARCHITECTURE)
        toml = VALID_MODULES_TOML.replace('path = "widget"', 'path = "a/b/c"').replace(
            'card = "widget/AGENTS.md"', 'card = "a/b/c/AGENTS.md"'
        )
        _write(os.path.join(root, 'modules.toml'), toml)
        _write(os.path.join(root, 'a', '__init__.py'), '')  # the anomaly: a package ABOVE the gap
        # deliberately no a/b/__init__.py -- the gap
        _write(os.path.join(root, 'a', 'b', 'c', 'AGENTS.md'), VALID_AGENTS_MD)
        _write(os.path.join(root, 'a', 'b', 'c', 'README.md'), VALID_README)
        _write(os.path.join(root, 'a', 'b', 'c', '__init__.py'), VALID_INIT.replace('widget.core', 'a.b.c.core'))
        _write(os.path.join(root, 'a', 'b', 'c', 'core.py'), VALID_CORE)
        fails, _ = sop_check.check(root, write=False)  # must not raise
        self.assertTrue(
            any('a/b/c: ancestor chain has a gap' in f for f in fails),
            f'expected a loud "ancestor chain has a gap" failure, got: {fails}',
        )


CONTAINER_WIDGET_MODULES_TOML = """[project]
name = "fixture"
line_cap_soft = 300
line_cap_hard = 500
architecture_word_budget = 1500

[modules.widget]
path = "container/widget"
owns = "test fixture code"
does_not_own = "everything else"
depends_on = []
public = ["foo", "BAR"]
card = "container/widget/AGENTS.md"
"""


class TestRelativeImports(_FixtureRepo):
    """Regression coverage for relative imports in the import-contract scan (check 5c):
    the gap was that a
    relative import (`from .x import y` / `from ..x import y` / `from . import x`) was invisible
    to check 5c, which only ever compared an ABSOLUTE dotted node.module against a registered
    module's dotted name -- exactly the shape of a package-internal import (`from
    .inner.screen import Thing` inside pkg/routes.py, whose own package
    `pkg` is the PARENT of the registered module `pkg.inner`). Reproduced
    here with `widget` nested at `container/widget` so `container` (a plain, unregistered package
    -- legal under the module rules) plays pkg's role."""

    def _container_repo(self):
        root = self._tmp
        _write(os.path.join(root, 'ARCHITECTURE.md'), VALID_ARCHITECTURE)
        _write(os.path.join(root, 'modules.toml'), CONTAINER_WIDGET_MODULES_TOML)
        _write(os.path.join(root, 'container', '__init__.py'), '')
        _write(os.path.join(root, 'container', 'widget', 'AGENTS.md'), VALID_AGENTS_MD)
        _write(os.path.join(root, 'container', 'widget', 'README.md'), VALID_README)
        _write(os.path.join(root, 'container', 'widget', '__init__.py'), VALID_INIT)
        _write(os.path.join(root, 'container', 'widget', 'core.py'), VALID_CORE)
        return root

    def test_single_dot_bypass(self):
        """from .widget.core import foo, inside container/legacy.py (container's own package)."""
        root = self._container_repo()
        _write(os.path.join(root, 'container', 'legacy.py'), "from .widget.core import foo\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("container/legacy.py imports container.widget.core, reaching past widget's package entry" in f
                for f in fails),
            f'expected a single-dot relative-import bypass failure, got: {fails}',
        )

    def test_double_dot_bypass(self):
        """from ..widget.core import foo, inside container/sub/legacy2.py (container.sub package,
        one level deeper -- '..' climbs back up to container)."""
        root = self._container_repo()
        _write(os.path.join(root, 'container', 'sub', '__init__.py'), '')
        _write(os.path.join(root, 'container', 'sub', 'legacy2.py'), "from ..widget.core import foo\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("container/sub/legacy2.py imports container.widget.core, reaching past widget's package entry" in f
                for f in fails),
            f'expected a double-dot relative-import bypass failure, got: {fails}',
        )

    def test_relative_from_package_import_submodule_name(self):
        """from .widget import core -- naming the internal file as the imported symbol, the relative
        form of the existing absolute test_form4."""
        root = self._container_repo()
        _write(os.path.join(root, 'container', 'legacy.py'), "from .widget import core\n")
        fails, _ = sop_check.check(root, write=False)
        self.assertTrue(
            any("container/legacy.py does `from container.widget import core`, reaching past widget's "
                "package entry" in f for f in fails),
            f'expected a relative from-package-import-submodule violation, got: {fails}',
        )

    def test_clean_arm_relative_import_through_the_entry_point_passes(self):
        """from .widget import foo -- the PUBLIC re-exported name, never reaching into core.py --
        must not be flagged."""
        root = self._container_repo()
        _write(os.path.join(root, 'container', 'legacy.py'), "from .widget import foo\n\nfoo()\n")
        sop_check.check(root, write=True)
        fails, _ = sop_check.check(root, write=False)
        self.assertEqual(
            fails, [], f'a legacy file using the relative entry point correctly must not be flagged, got: {fails}'
        )

    def test_no_init_py_importer_fails_loudly(self):
        """An importer with NO enclosing __init__.py doing a relative import is a named, loud FAIL --
        never a silent skip or a crash (the same rule already governing dotted_import_name's own
        ancestor-gap case)."""
        root = self._container_repo()
        _write(os.path.join(root, 'orphan', 'leg.py'), "from .x import y\n")  # deliberately NO orphan/__init__.py
        fails, _ = sop_check.check(root, write=False)  # must not raise
        self.assertTrue(
            any('orphan/leg.py:1: relative import cannot be resolved -- orphan: not a package (no __init__.py)' in f
                for f in fails),
            f'expected a loud no-__init__.py failure, got: {fails}',
        )

    def test_out_of_range_level_fails_loudly(self):
        """A relative import whose level runs out of package components (more dots than the
        importer's own package has ancestors) is a named, loud FAIL -- never a silent skip or crash."""
        root = self._container_repo()
        # container is a single-component package; three dots needs two ancestors above it -- none exist.
        _write(os.path.join(root, 'container', 'deep_legacy.py'), "from ...x import y\n")
        fails, _ = sop_check.check(root, write=False)  # must not raise
        self.assertTrue(
            any('container/deep_legacy.py:1: relative import "from ...x import ..." could not be resolved' in f
                for f in fails),
            f'expected a loud out-of-range-level failure, got: {fails}',
        )


if __name__ == '__main__':
    unittest.main()
