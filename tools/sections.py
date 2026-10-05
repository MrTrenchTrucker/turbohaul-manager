"""Shared Markdown section extraction for the module-structure tools (one implementation, reused by build + verify).

Vendored verbatim from the upstream module-structure tools. No adaptation needed: generic text
utility, nothing here assumes a fully-modularized repo.
"""
import re


def section(text, heading, stop_heading=None):
    """Return the section starting at the line `heading` up to (not including) `stop_heading`, or to EOF.

    Raises ValueError if a heading is missing, so a renamed section fails loudly instead of returning nothing.
    """
    start = text.find('\n' + heading + '\n')
    if start < 0:
        raise ValueError(f'heading not found: {heading!r}')
    start += 1
    if stop_heading is None:
        end = len(text)
    else:
        end = text.find('\n' + stop_heading + '\n', start)
        if end < 0:
            raise ValueError(f'stop heading not found: {stop_heading!r}')
        end += 1
    return text[start:end].rstrip('\n') + '\n'


def sop_version(text):
    """Return the '**Version:** x.y.z' value of an SOP, or raise ValueError."""
    m = re.search(r'\*\*Version:\*\* ([0-9][0-9.]*)', text)
    if not m:
        raise ValueError('no **Version:** line')
    return m.group(1)
