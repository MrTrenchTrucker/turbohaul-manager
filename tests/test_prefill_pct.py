"""Telemetry: /status.generation `prefill_pct` field.

`prefill_pct` = round(n_prompt_tokens_processed / n_prompt_tokens * 100) while a
slot is_processing and n_prompt_tokens > 0, else null. Distinct from the existing
`prompt_progress` (a fraction shown only during the prefill state). Guards
divide-by-zero on n_prompt_tokens.
"""
from unittest.mock import MagicMock

from turbohaul.live_monitor import LiveSlotsPoller, idle_generation


def _poller() -> LiveSlotsPoller:
    # _compute/_derive never touch self._mgr; a MagicMock ctor arg is enough
    # (the httpx.AsyncClient built in __init__ makes no network call here).
    return LiveSlotsPoller(mgr=MagicMock())


def _slot(*, is_processing, n_prompt, n_prompt_proc, n_decoded):
    return {
        "id_task": 1,
        "is_processing": is_processing,
        "n_prompt_tokens": n_prompt,
        "n_prompt_tokens_processed": n_prompt_proc,
        "n_ctx": 4096,
        "next_token": [{
            "n_decoded": n_decoded,
            "n_remain": 100,
            "has_next_token": True,
        }],
        "params": {"max_tokens": 128, "stream": False},
    }


def test_prefill_pct_during_prefill():
    # mid-prefill: 50 of 200 prompt tokens processed, no output yet.
    gen = _poller()._compute(
        [_slot(is_processing=True, n_prompt=200, n_prompt_proc=50, n_decoded=0)],
        resp_t=1.0, pid=123, spawn_seq=1, thread_or_slot="t",
    )
    assert gen["prefill_pct"] == 25
    # distinct from the existing fractional prompt_progress
    assert gen["prompt_progress"] == 0.25
    assert gen["prefill_pct"] != gen["prompt_progress"]


def test_prefill_pct_100_when_prefill_complete_and_generating():
    gen = _poller()._compute(
        [_slot(is_processing=True, n_prompt=200, n_prompt_proc=200, n_decoded=40)],
        resp_t=1.0, pid=123, spawn_seq=1, thread_or_slot="t",
    )
    assert gen["prefill_pct"] == 100


def test_prefill_pct_null_when_not_processing():
    gen = _poller()._compute(
        [_slot(is_processing=False, n_prompt=200, n_prompt_proc=50, n_decoded=0)],
        resp_t=1.0, pid=123, spawn_seq=1, thread_or_slot="t",
    )
    assert gen["state"] == "idle"
    assert gen["prefill_pct"] is None


def test_prefill_pct_null_and_no_zerodiv_when_n_prompt_zero():
    # n_prompt_tokens == 0 must yield null, NOT ZeroDivisionError.
    gen = _poller()._compute(
        [_slot(is_processing=True, n_prompt=0, n_prompt_proc=0, n_decoded=5)],
        resp_t=1.0, pid=123, spawn_seq=1, thread_or_slot="t",
    )
    assert gen["prefill_pct"] is None


def test_idle_generation_carries_null_prefill_pct():
    assert idle_generation()["prefill_pct"] is None
