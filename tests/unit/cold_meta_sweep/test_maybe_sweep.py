"""maybe_sweep: the decision whether a pass is due, the counters, the thread hop and the log line."""
import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from turbohaul.cold_meta_sweep import maybe_sweep

LOGGER = "turbohaul.manager"
IDLE = 7


def _counts(**over):
    base = {"stripped": 0, "unchanged": 0, "invalid": 0, "missing": 0, "error": 0, "rows": 0, "pass_done": True}
    base.update(over)
    return base


class FakeHost:
    _COLD_META_IDLE_AFTER_PASS_S = IDLE

    def __init__(self, age_hours=2, counts=None):
        self.runtime = SimpleNamespace(kv=SimpleNamespace(cold_meta_purge_age_hours=age_hours))
        self._cold_meta_next_pass_monotonic = 0.0
        self._cold_meta_stripped_lifetime = 0
        self._cold_meta_invalid_lifetime = 0
        self.counts = counts if counts is not None else _counts()
        self.ran_with = []
        self.raises = None

    def _cold_meta_sweep_once(self, age_hours):
        self.ran_with.append(age_hours)
        if self.raises:
            raise self.raises
        return self.counts


def _run(host):
    return asyncio.run(maybe_sweep(host))


def _lines(caplog):
    return [r for r in caplog.records if r.name == LOGGER]


@pytest.fixture
def caplog_all(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    return caplog


def test_age_zero_and_negative_mean_off_and_nothing_happens(caplog_all):
    for age in (0, 0.0, -1):
        host = FakeHost(age_hours=age)
        assert _run(host) is None
        assert host.ran_with == []
        assert host._cold_meta_next_pass_monotonic == 0.0
    assert _lines(caplog_all) == []


def test_a_run_is_skipped_while_the_idle_timer_has_not_passed(caplog_all):
    host = FakeHost()
    host._cold_meta_next_pass_monotonic = time.monotonic() + 3600
    timer = host._cold_meta_next_pass_monotonic
    assert _run(host) is None
    assert host.ran_with == []
    assert host._cold_meta_next_pass_monotonic == timer
    assert _lines(caplog_all) == []


def test_a_due_run_returns_the_counts_of_the_run_and_passes_the_age_as_a_float():
    host = FakeHost(age_hours=2, counts=_counts(rows=3, stripped=3))
    result = _run(host)
    assert result is host.counts
    assert host.ran_with == [2.0] and isinstance(host.ran_with[0], float)


def test_the_age_is_read_on_every_call():
    host = FakeHost(age_hours=1)
    _run(host)
    host.runtime.kv.cold_meta_purge_age_hours = 5
    host._cold_meta_next_pass_monotonic = 0.0
    _run(host)
    assert host.ran_with == [1.0, 5.0]


def test_the_run_is_handed_to_the_thread_hop_as_the_hosts_own_bound_method(monkeypatch):
    host = FakeHost()
    handed = []
    real = asyncio.to_thread

    async def spy(fn, *args, **kwargs):
        handed.append((fn, args))
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy)
    _run(host)
    assert len(handed) == 1
    fn, args = handed[0]
    assert fn == host._cold_meta_sweep_once and fn.__self__ is host
    assert fn.__name__ == "_cold_meta_sweep_once"
    assert args == (2.0,)


def test_a_finished_pass_sets_the_idle_timer_and_an_unfinished_one_does_not():
    host = FakeHost(counts=_counts(pass_done=True))
    before = time.monotonic()
    _run(host)
    after = time.monotonic()
    assert before + IDLE <= host._cold_meta_next_pass_monotonic <= after + IDLE
    partial = FakeHost(counts=_counts(pass_done=False, rows=5, stripped=5))
    _run(partial)
    assert partial._cold_meta_next_pass_monotonic == 0.0


def test_the_lifetime_counters_accumulate_stripped_and_invalid_only():
    host = FakeHost(counts=_counts(rows=9, stripped=4, invalid=2, unchanged=2, error=1))
    _run(host)
    assert (host._cold_meta_stripped_lifetime, host._cold_meta_invalid_lifetime) == (4, 2)
    host._cold_meta_next_pass_monotonic = 0.0
    _run(host)
    assert (host._cold_meta_stripped_lifetime, host._cold_meta_invalid_lifetime) == (8, 4)


def test_stripped_rows_alone_log_one_info_line_with_the_fixed_text(caplog_all):
    host = FakeHost(age_hours=2, counts=_counts(rows=3, stripped=2, unchanged=1))
    _run(host)
    recs = _lines(caplog_all)
    assert len(recs) == 1
    assert recs[0].levelno == logging.INFO
    assert recs[0].getMessage() == (
        "cold slot metadata sweep: rows=3 stripped=2 unchanged=1 invalid=0 error=0 (age_hours=2.0, pass_done=True)")


def test_an_invalid_row_logs_a_warning(caplog_all):
    host = FakeHost(age_hours=1.5, counts=_counts(rows=2, stripped=1, invalid=1, pass_done=False))
    _run(host)
    recs = _lines(caplog_all)
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    assert recs[0].getMessage() == (
        "cold slot metadata sweep: rows=2 stripped=1 unchanged=0 invalid=1 error=0 (age_hours=1.5, pass_done=False)")


def test_a_failed_row_alone_logs_a_warning(caplog_all):
    host = FakeHost(counts=_counts(rows=1, error=1))
    _run(host)
    recs = _lines(caplog_all)
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    assert recs[0].getMessage() == (
        "cold slot metadata sweep: rows=1 stripped=0 unchanged=0 invalid=0 error=1 (age_hours=2.0, pass_done=True)")


def test_nothing_stripped_invalid_or_failed_logs_nothing(caplog_all):
    host = FakeHost(counts=_counts(rows=4, unchanged=3, missing=1))
    _run(host)
    assert _lines(caplog_all) == []


def test_a_failing_run_propagates_and_leaves_the_timer_and_counters_alone(caplog_all):
    host = FakeHost()
    host.raises = RuntimeError("run failed")
    with pytest.raises(RuntimeError, match="run failed"):
        _run(host)
    assert host._cold_meta_next_pass_monotonic == 0.0
    assert (host._cold_meta_stripped_lifetime, host._cold_meta_invalid_lifetime) == (0, 0)
    assert _lines(caplog_all) == []
