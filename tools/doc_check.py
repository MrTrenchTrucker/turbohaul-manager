#!/usr/bin/env python3
"""Module docs checks.

Every module, at every level, carries a card (`AGENTS.md`) and a `README.md`. The card must agree with the
registry, and a code module's registry `public` list must be its real public functions. A parent's README names
each of its sub-modules. And a module whose code changed since the base commit must have changed its README too,
unless a README waiver is honoured for it (only in a markdown file under a workorders/ directory). Modules marked `docs_pending` in the registry are skipped
(until their docs are written).

Vendored from upstream module-structure tools and adapted: `_code_public` now
resolves a package's real public surface from its `__init__.py`'s `__all__` when one is present, as the bare
names `__all__` actually exports (which internal file defines a name is invisible to a caller through the
entry point); a flat module with no `__all__` keeps the upstream `file.name`-qualified heuristic. Either way
the AST walk now also sees module-level constants (`Assign`/`AnnAssign` targets, e.g. `LIMITS`), not just
`FunctionDef`/`ClassDef` -- the upstream walk silently missed every constant export.

Usage: doc_check.py [--base REF]   exit 1 on any finding. The base defaults to $SOP_BASE, then the merge-base
with main. The base must be the commit the change started from: on a branch carrying several changes, set
SOP_BASE per change, or a later change rides on an earlier change's README edit.
"""
import argparse
import ast
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sections import section  # noqa: E402

WAIVER = re.compile(r'README waiver:\s*`?([A-Za-z0-9_]+)`?\s*:\s*\S')
CLI_ENTRY = 'main'


def _all_list(init_path):
    """The literal `__all__` list in an __init__.py, or None if absent or not a literal list."""
    if not os.path.isfile(init_path):
        return None
    for node in ast.parse(open(init_path).read()).body:
        targets = node.targets if isinstance(node, ast.Assign) else \
            ([node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == '__all__' and isinstance(node.value, (ast.List, ast.Tuple)):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


def _top_level_defined(base, fn):
    """Names this file defines at module level -- functions, classes, AND constants
    (Assign/AnnAssign targets). The upstream walk only ever saw FunctionDef/ClassDef."""
    names = set()
    for node in ast.parse(open(os.path.join(base, fn)).read()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _init_imported_names(init_path):
    """Names `__init__.py` imports from its OWN submodules (`from .x import Y`),
    whether or not they end up in `__all__`. A name reviewed and imported here but deliberately left
    out of `__all__` (e.g. a class re-exported only for internal typing use, `# noqa: F401`) was a
    reviewed choice, not a silent leak -- see `_undeclared_public_names`."""
    if not os.path.isfile(init_path):
        return set()
    names = set()
    for node in ast.walk(ast.parse(open(init_path).read())):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _undeclared_public_names(root, m):
    """`_code_public`'s package-shape branch returns `__all__ ∩ defined` -- so a
    public-looking (no leading underscore) top-level name that a module's author simply forgot to add
    to `__all__` is invisible to it, and gets no finding at all. This catches that gap directly: a
    name defined in one of the module's OWN files, public-looking, that `__init__.py` never even
    imports (not in `__all__`, not imported at all) is a silent leak nobody reviewed. A name
    `__init__.py` DOES import but leaves out of `__all__` on purpose is not this -- it was reviewed;
    see `_init_imported_names`. Only applies to package-shape modules (an `__all__` present); a flat
    module's every public name is already visible to `_code_public` directly."""
    base = os.path.join(root, m['path'])
    init_path = os.path.join(base, '__init__.py')
    all_list = _all_list(init_path)
    if all_list is None:
        return {}
    defined_in = {}
    for fn in sorted(os.listdir(base)):
        if fn.endswith('.py') and fn != '__init__.py':
            for nm in _top_level_defined(base, fn):
                defined_in.setdefault(nm, fn[:-3])
    imported = _init_imported_names(init_path)
    all_set = set(all_list)
    return {nm: stem for nm, stem in defined_in.items()
            if not nm.startswith('_') and nm not in all_set and nm not in imported}


def _code_public(root, m):
    """Public surface of a code module, in the same shape the registry names it.

    If the module is a package (an __init__.py with a literal `__all__`), the entry file is the
    ONLY way consumers reach it (the module import contract), so its public names are the bare
    `__all__` names themselves -- which internal file happens to define a name is invisible to a caller and
    must not be. Without an `__all__` (a flat module, e.g. a plain scripts folder), the upstream `file.name`-qualified
    heuristic applies, extended to constants (Assign/AnnAssign) as well as functions/classes.
    """
    base = os.path.join(root, m['path'])
    defined_in = {}
    for fn in sorted(os.listdir(base)):
        if fn.endswith('.py'):
            for nm in _top_level_defined(base, fn):
                defined_in.setdefault(nm, fn[:-3])
    all_list = _all_list(os.path.join(base, '__init__.py'))
    if all_list is not None:
        return {n for n in all_list if n in defined_in}
    return {f'{stem}.{nm}' for nm, stem in defined_in.items() if not nm.startswith('_')}


def _artifacts(root, m):
    """What a non-code module can expose: its file names and the skill names in its SKILL.md files."""
    found = set()
    for dp, _dn, files in os.walk(os.path.join(root, m['path'])):
        for fn in files:
            found.add(fn)
            if fn == 'SKILL.md':
                hit = re.search(r'^name:\s*(\S+)', open(os.path.join(dp, fn)).read(4000), re.M)
                if hit:
                    found.add(hit.group(1))
    return found


def _ticked(text):
    return {t.split('(')[0].strip() for t in re.findall(r'`([^`\n]+)`', text)}


def card_findings(root, reg):
    """Static checks: README present, card matches registry and code, parent README names its sub-modules."""
    mods, out = reg['modules'], []
    by_path = {m['path']: n for n, m in mods.items()}
    for name, m in mods.items():
        if m.get('docs_pending'):
            continue
        p = m['path']
        if not os.path.isfile(os.path.join(root, p, 'README.md')):
            out.append(f'{name}: {p}/README.md is missing. Every module, at every level, needs one: what it does and how it '
                       f'works, in plain words.')
        card_path = os.path.join(root, m['card'])
        if not os.path.isfile(card_path):
            continue  # sop_check reports a missing card
        card = open(card_path).read()
        try:
            iface, deps_text = section(card, '## Public Interface', '## Depends On'), section(card, '## Depends On', '## Invariants')
        except ValueError:
            continue  # sop_check reports a missing card section
        public, code = set(m.get('public', [])), _code_public(root, m)
        for nm, stem in sorted(_undeclared_public_names(root, m).items()):
            out.append(f'{name}: `{nm}` ({stem}.py) looks public (no leading underscore) but '
                       f'__init__.py never imports it -- neither in __all__ nor imported at all. Add '
                       f'it to __all__ (and the registry) if it is meant to be public, or rename it '
                       f'with a leading underscore if it is not.')
        exposable = code if code else _artifacts(root, m)
        listed = {t for t in _ticked(iface) if t in exposable or t in public}
        for t in sorted(public - listed):
            out.append(f'{name}: the registry lists `{t}` as public, but the card\'s Public Interface does not. Card and '
                       f'registry must say the same thing; reread {m["card"]}.')
        for t in sorted(listed - public):
            out.append(f'{name}: the card\'s Public Interface lists `{t}`, but the registry\'s public list does not. Card and '
                       f'registry must say the same thing; an interface change needs approval from the module owner.')
        if code:
            for t in sorted(code - public):
                # A bare (package-shape) name has no file qualifier to split -- the CLI_ENTRY
                # convention only ever applied to flat, file.name-qualified modules.
                if '.' not in t or t.split('.', 1)[1] != CLI_ENTRY:
                    out.append(f'{name}: `{t}` is public in the code but not in the registry\'s public list. Name it private '
                               f'(leading underscore) or ask the module owner to add it to the interface.')
            for t in sorted(public - code):
                out.append(f'{name}: the registry lists `{t}` as public, but the code has no such public function or class. '
                           f'The interface changed without the card and registry; reread {m["card"]}.')
        else:
            for t in sorted(public - exposable):
                out.append(f'{name}: the registry lists `{t}` as public, but nothing in {p}/ provides it.')
        named = {n for n in mods if n != name and re.search(rf'(?<![\w-]){re.escape(n)}(?![\w-])', deps_text)}
        if named != set(m.get('depends_on', [])):
            out.append(f'{name}: the card\'s Depends On names {sorted(named)}, but the registry says '
                       f'{sorted(m.get("depends_on", []))}. Card and registry must say the same thing.')
        parts = p.split('/')
        for i in range(len(parts) - 1, 0, -1):
            parent = by_path.get('/'.join(parts[:i]))
            if parent:
                readme = os.path.join(root, mods[parent]['path'], 'README.md')
                if not mods[parent].get('docs_pending') and os.path.isfile(readme) and \
                        not re.search(rf'(?<![\w-]){re.escape(name)}(?![\w-])', open(readme).read()):
                    out.append(f'{parent}: its README does not name its sub-module `{name}`. A parent\'s README names each of '
                               f'its sub-modules; update {mods[parent]["path"]}/README.md.')
                break
    return out


def resolve_base(root):
    """The commit the work started from: $SOP_BASE, else the merge-base of HEAD with main. Raises if neither resolves."""
    if os.environ.get('SOP_BASE'):
        return os.environ['SOP_BASE']
    for ref in ('origin/main', 'main'):
        r = subprocess.run(['git', '-C', root, 'merge-base', 'HEAD', ref], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise RuntimeError('cannot find the commit your work started from: set SOP_BASE to the base commit of your change')


def _changed_files(root, base):
    """Files changed since `base`, including uncommitted edits and new untracked files."""
    def git(*a):
        return subprocess.run(['git', '-C', root, *a], capture_output=True, text=True, check=True).stdout.split('\n')
    return {f for f in set(git('diff', '--name-only', base)) | set(git('ls-files', '--others', '--exclude-standard')) if f}


def freshness_findings(root, reg, base):
    """A module whose code changed since `base` must also have changed its README, or carry a README waiver."""
    mods, changed = reg['modules'], _changed_files(root, base)
    waivers = set()
    for f in changed:
        if f.startswith('workorders/') and f.endswith('.md') and os.path.isfile(os.path.join(root, f)):
            waivers |= set(WAIVER.findall(open(os.path.join(root, f)).read()))
    code, readme = set(), set()
    for f in changed:
        owners = [n for n, m in mods.items() if f.startswith(m['path'].rstrip('/') + '/')]
        if not owners:
            continue
        owner = max(owners, key=lambda n: len(mods[n]['path']))
        rest = f[len(mods[owner]['path'].rstrip('/')) + 1:]
        if rest == 'README.md':
            readme.add(owner)
        elif rest != 'AGENTS.md':
            code.add(owner)
    out = []
    for name in sorted(code - readme - waivers):
        if mods[name].get('docs_pending'):
            continue
        p = mods[name]['path']
        out.append(f'{name}: you changed code in {p}/ but not {p}/README.md. Reread {p}/README.md and {mods[name]["card"]}, '
                   f'check your change against them, and update the README in this same change. If it truly needs no change, '
                   f'add "README waiver: {name}: <why>" to your change notes for the reviewer to accept.')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--base', help='commit to compare against (default: $SOP_BASE, then the merge-base with main)')
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    import tomllib
    reg = tomllib.load(open(os.path.join(a.root, 'modules.toml'), 'rb'))
    found = card_findings(a.root, reg) + freshness_findings(a.root, reg, a.base or resolve_base(a.root))
    print('\n'.join(found) if found else 'module docs: every card and README present, matching, and current')
    print(f'RESULT {"FAIL (" + str(len(found)) + ")" if found else "PASS"}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
