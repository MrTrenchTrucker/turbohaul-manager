"""Tests for tools/doc_check.py's own functions:
card_findings/_code_public's __all__-visibility handling and freshness_findings/resolve_base/
_changed_files. Every fixture is built from scratch in a tempdir;
nothing here touches the real repository tree. The freshness arms are the only ones that need a real (throwaway,
hermetic) git repo -- freshness_findings/_changed_files/resolve_base all shell out to git directly, so
there is no way to test them without one; no network, no shared state, no writes outside the tempdir.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import doc_check  # noqa: E402

WIDGET_AGENTS_MD = """# widget

## Purpose
A test fixture module.

## Owns
Test fixture code.

## Does Not Own
Everything else.

## Public Interface
- `foo()`: a trivial function.

## Depends On
- none

## Invariants
- none

## Test Locations
- tools/tests/test_doc_check.py (this file, as a fixture consumer)

## Known Gotchas
- none
"""

WIDGET_README = "# widget\n\nA test fixture module for the doc_check red-arm proof.\n"


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def _reg(public=("foo",)):
    return {'modules': {
        'widget': {
            'path': 'widget',
            'owns': 'test fixture code',
            'does_not_own': 'everything else',
            'depends_on': [],
            'public': list(public),
            'card': 'widget/AGENTS.md',
        }
    }}


def _git(root, *args):
    return subprocess.run(['git', '-C', root, *args], capture_output=True, text=True, check=True)


class _FixtureRepo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='doc_check_fixture_')
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)

    def _write_widget(self, init_body, core_body, agents_md=WIDGET_AGENTS_MD, readme=WIDGET_README):
        root = self._tmp
        _write(os.path.join(root, 'widget', 'AGENTS.md'), agents_md)
        _write(os.path.join(root, 'widget', 'README.md'), readme)
        _write(os.path.join(root, 'widget', '__init__.py'), init_body)
        _write(os.path.join(root, 'widget', 'core.py'), core_body)
        return root


# --- A public-looking name left out of __all__ gets no finding from
#     _code_public's __all__-intersection (it only ever reports names __all__ already claims).
#     _undeclared_public_names / card_findings must catch it -- but not a name __init__.py imports on
#     purpose and just leaves out of __all__ (a documented, deliberate shape,
#     not a leak).

class TestUndeclaredPublicNames(_FixtureRepo):

    def test_red_public_name_not_in_all_and_not_imported_is_flagged(self):
        root = self._write_widget(
            init_body='from widget.core import foo\n\n__all__ = ["foo"]\n',
            core_body='def foo():\n    return 1\n\n\ndef bar():\n    return 2\n',  # bar: the leak
        )
        out = doc_check.card_findings(root, _reg())
        self.assertTrue(
            any('`bar` (core.py) looks public' in f and f.startswith('widget:') for f in out),
            f'expected a finding naming the undeclared public name bar, got: {out}',
        )

    def test_clean_name_imported_but_deliberately_excluded_from_all_is_not_flagged(self):
        """A documented package shape: __init__.py imports a name for internal use but leaves
        it out of __all__ on purpose -- that was a deliberate choice, not a silent leak, must not be flagged."""
        root = self._write_widget(
            init_body='from widget.core import foo, Helper  # noqa: F401\n\n__all__ = ["foo"]\n',
            core_body='def foo():\n    return 1\n\n\nclass Helper:\n    pass\n',
        )
        out = doc_check.card_findings(root, _reg())
        self.assertFalse(
            any('Helper' in f for f in out),
            f'a name __init__.py imports on purpose (just left out of __all__) must not be flagged, got: {out}',
        )

    def test_clean_name_in_all_is_not_flagged(self):
        root = self._write_widget(
            init_body='from widget.core import foo, bar\n\n__all__ = ["foo", "bar"]\n',
            core_body='def foo():\n    return 1\n\n\ndef bar():\n    return 2\n',
            agents_md=WIDGET_AGENTS_MD.replace(
                '- `foo()`: a trivial function.',
                '- `foo()`: a trivial function.\n- `bar()`: another trivial function.',
            ),
        )
        out = doc_check.card_findings(root, _reg(public=("foo", "bar")))
        self.assertFalse(any('looks public' in f for f in out), f'expected no undeclared-public-name finding, got: {out}')

    def test_clean_private_name_not_flagged(self):
        root = self._write_widget(
            init_body='from widget.core import foo\n\n__all__ = ["foo"]\n',
            core_body='def foo():\n    return 1\n\n\ndef _helper():\n    return 2\n',
        )
        out = doc_check.card_findings(root, _reg())
        self.assertFalse(
            any('looks public' in f for f in out),
            f'a leading-underscore name must never be flagged, got: {out}',
        )

    def test_flat_module_unaffected_no_all(self):
        """A flat module (no __init__.py __all__) is untouched by this check -- _code_public already
        sees every one of its public names directly; _undeclared_public_names must return nothing."""
        root = self._tmp
        out = doc_check._undeclared_public_names(root, {'path': 'widget'})
        self.assertEqual(out, {}, f'a module with no __all__ must get no undeclared-public-name findings, got: {out}')


# --- freshness_findings/resolve_base/_changed_files tests.

@unittest.skipUnless(shutil.which('git'), 'needs git')
class TestFreshness(_FixtureRepo):

    def _init_git_repo(self):
        root = self._tmp
        _git(root, 'init', '-q', '-b', 'main')
        _git(root, 'config', 'user.email', 'test@example.invalid')
        _git(root, 'config', 'user.name', 'Test')
        return root

    def _commit_base(self, root):
        self._write_widget(
            init_body='from widget.core import foo\n\n__all__ = ["foo"]\n',
            core_body='def foo():\n    return 1\n',
        )
        _git(root, 'add', '-A')
        _git(root, 'commit', '-q', '-m', 'base')
        return _git(root, 'rev-parse', 'HEAD').stdout.strip()

    def test_code_changed_without_readme_fails(self):
        root = self._init_git_repo()
        base = self._commit_base(root)
        _write(os.path.join(root, 'widget', 'core.py'), 'def foo():\n    return 2\n')  # code only
        out = doc_check.freshness_findings(root, _reg(), base)
        self.assertTrue(
            any(f.startswith('widget:') and 'README.md' in f for f in out),
            f'expected a code-changed-without-README finding, got: {out}',
        )

    def test_readme_waiver_suppresses_the_finding(self):
        root = self._init_git_repo()
        base = self._commit_base(root)
        _write(os.path.join(root, 'widget', 'core.py'), 'def foo():\n    return 2\n')
        _write(os.path.join(root, 'workorders', 'WO-TEST.md'),
               'README waiver: widget: cosmetic-only change, no behaviour or interface change.\n')
        out = doc_check.freshness_findings(root, _reg(), base)
        self.assertFalse(
            any(f.startswith('widget:') and 'README.md' in f for f in out),
            f'a README waiver in a workorders/*.md file must suppress the finding, got: {out}',
        )

    def test_readme_changed_alongside_code_does_not_fail(self):
        root = self._init_git_repo()
        base = self._commit_base(root)
        _write(os.path.join(root, 'widget', 'core.py'), 'def foo():\n    return 2\n')
        _write(os.path.join(root, 'widget', 'README.md'), WIDGET_README + '\nUpdated for the new foo().\n')
        out = doc_check.freshness_findings(root, _reg(), base)
        self.assertEqual(out, [], f'README updated alongside code -- expected zero findings, got: {out}')

    def test_uncommitted_and_untracked_changes_are_seen(self):
        """_changed_files must see BOTH uncommitted edits to tracked files and brand-new untracked
        files, not just committed diffs -- a work-in-progress change is exactly this shape."""
        root = self._init_git_repo()
        base = self._commit_base(root)
        _write(os.path.join(root, 'widget', 'core.py'), 'def foo():\n    return 2\n')  # uncommitted edit
        changed = doc_check._changed_files(root, base)
        self.assertIn('widget/core.py', changed)

    def test_resolve_base_reads_sop_base_env_var(self):
        os.environ['SOP_BASE'] = 'deadbeef'
        try:
            self.assertEqual(doc_check.resolve_base(self._tmp), 'deadbeef')
        finally:
            del os.environ['SOP_BASE']

    def test_resolve_base_falls_back_to_merge_base_with_main(self):
        os.environ.pop('SOP_BASE', None)
        root = self._init_git_repo()
        base = self._commit_base(root)
        _git(root, 'checkout', '-q', '-b', 'feature')
        _write(os.path.join(root, 'widget', 'core.py'), 'def foo():\n    return 3\n')
        _git(root, 'add', '-A')
        _git(root, 'commit', '-q', '-m', 'feature work')
        self.assertEqual(
            doc_check.resolve_base(root), base,
            'with no SOP_BASE set, resolve_base must fall back to the merge-base with main',
        )


if __name__ == '__main__':
    unittest.main()
