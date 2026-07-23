"""Prefill_pct denominator fix — live engine behavior.

The engine streams hd["n_prompt"] as the RUNNING processed-count
(= n_prompt_proc + n_prompt_cache), so the OLD calculation
(pref_num / n_prompt) always yielded 100% during prefill.

The fix threads admission_ctx_len (TRUE total prompt tokens known
at request admission) through the headline into _derive, and uses
it as the denominator. Tests here verify the fix against LIVE
engine behavior where n_prompt is the running count.
"""

from turbohaul.live_monitor import LiveSlotsPoller

# ── minimal duck-type stubs ──────────────────────────────────────

class _FakeHandle:
    def __init__(self, pid=123, port=11500):
        self.pid = pid
        self.port = port

class _FakeSlot:
    def __init__(self, state_value="ACTIVE", slot_id="s-1",
                 thread_id="t-1", admission_ctx_len=1000):
        self.state = type("S", (), {"value": state_value})()
        self.slot_id = slot_id
        self.thread_id = thread_id
        self.admission_ctx_len = admission_ctx_len

class _FakeMgr:
    def __init__(self):
        self._active_slot = _FakeSlot(admission_ctx_len=1000)
        self._active_handle = _FakeHandle()
        self._stop_event = type("E", (), {"is_set": lambda: False})()
        self.live_generation = None
        self.live_generations = {}
        self._vram_free_mib = None
        self._vram_total_mib = None

    def _active_spawn_seq(self):
        return 1


# ── tests ───────────────────────────────────────────────────────

class TestPrefillPctDenominator:
    """prefill_pct uses admission_ctx_len, not running n_prompt."""

    def test_prefill_pct_is_zero_at_start_of_prefill(self):
        """When proc=0, cache=0, prefill_pct should be 0% (not 100%)."""
        mgr = _FakeMgr()
        poller = LiveSlotsPoller(mgr)

        # Simulate /slots response: prefill just started
        slots_data = [{
            "is_processing": True,
            "id_task": 1,
            "params": {"max_tokens": 2048, "n_predict": 2048, "stream": True},
            "next_token": [{"n_decoded": 0, "n_remain": 2048, "has_next_token": True}],
            "n_prompt_tokens": 0,       # engine hasn't started counting yet
            "n_prompt_tokens_processed": 0,
            "n_prompt_tokens_cache": 0,
            "n_ctx": 32768,
        }]

        gen = poller._compute(
            slots_data, 0.0, 123, 1, "s-1", admission_ctx_len=1000
        )
        # At the very start, no tokens processed yet
        assert gen["prefill_pct"] == 0, \
            f"Expected 0% at prefill start, got {gen['prefill_pct']}"

    def test_prefill_pct_rises_during_prefill(self):
        """Live: n_prompt = running count = proc+cache. With OLD code,
        prefill_pct = (proc+cache)/(proc+cache) = 100 always.
        With the fix, prefill_pct = (proc+cache)/admission_ctx_len.

        Example: 4096 processed + 43421 cached = 47517 running n_prompt.
        admission_ctx_len = 50000 (true total).
        prefill_pct should be ~95%, not 100%.
        """
        mgr = _FakeMgr()
        poller = LiveSlotsPoller(mgr)

        slots_data = [{
            "is_processing": True,
            "id_task": 1,
            "params": {"max_tokens": 2048, "n_predict": 2048, "stream": True},
            "next_token": [{"n_decoded": 0, "n_remain": 2048, "has_next_token": True}],
            "n_prompt_tokens": 47517,       # running count = proc+cache
            "n_prompt_tokens_processed": 4096,
            "n_prompt_tokens_cache": 43421,
            "n_ctx": 32768,
        }]

        gen = poller._compute(
            slots_data, 0.0, 123, 1, "s-1", admission_ctx_len=50000
        )
        # OLD (broken): (4096+43421)/47517 = 100%
        # NEW (fixed):  (4096+43421)/50000 = 95%
        assert gen["prefill_pct"] == 95, \
            f"Expected 95%, got {gen['prefill_pct']} (prefill_pct fix regressed)"

    def test_prefill_pct_fallback_when_admission_ctx_len_zero(self):
        """When admission_ctx_len is 0 (older engine / test path),
        fall back to hd["n_prompt"] (the old behavior)."""
        mgr = _FakeMgr()
        poller = LiveSlotsPoller(mgr)

        slots_data = [{
            "is_processing": True,
            "id_task": 1,
            "params": {"max_tokens": 2048, "n_predict": 2048, "stream": True},
            "next_token": [{"n_decoded": 0, "n_remain": 2048, "has_next_token": True}],
            "n_prompt_tokens": 5000,
            "n_prompt_tokens_processed": 2500,
            "n_prompt_tokens_cache": 2500,
            "n_ctx": 32768,
        }]

        gen = poller._compute(
            slots_data, 0.0, 123, 1, "s-1", admission_ctx_len=0
        )
        # Fallback to old behavior: (2500+2500)/5000 = 100%
        assert gen["prefill_pct"] == 100

    def test_prefill_pct_clamped_at_100_when_complete(self):
        """When proc+cache >= total, prefill_pct is clamped to 100."""
        mgr = _FakeMgr()
        poller = LiveSlotsPoller(mgr)

        slots_data = [{
            "is_processing": True,
            "id_task": 1,
            "params": {"max_tokens": 2048, "n_predict": 2048, "stream": True},
            "next_token": [{"n_decoded": 0, "n_remain": 2048, "has_next_token": True}],
            "n_prompt_tokens": 10000,
            "n_prompt_tokens_processed": 7000,
            "n_prompt_tokens_cache": 4000,  # 7000+4000=11000 > 10000
            "n_ctx": 32768,
        }]

        gen = poller._compute(
            slots_data, 0.0, 123, 1, "s-1", admission_ctx_len=10000
        )
        assert gen["prefill_pct"] == 100, \
            f"Expected 100% (clamped), got {gen['prefill_pct']}"

    def test_prefill_pct_null_when_no_counters(self):
        """When proc is None and cache is None, prefill_pct is None."""
        mgr = _FakeMgr()
        poller = LiveSlotsPoller(mgr)

        slots_data = [{
            "is_processing": True,
            "id_task": 1,
            "params": {"max_tokens": 2048, "n_predict": 2048, "stream": True},
            "next_token": [{"n_decoded": 0, "n_remain": 2048, "has_next_token": True}],
            "n_prompt_tokens": 1000,
            "n_prompt_tokens_processed": None,
            "n_prompt_tokens_cache": None,
            "n_ctx": 32768,
        }]

        gen = poller._compute(
            slots_data, 0.0, 123, 1, "s-1", admission_ctx_len=1000
        )
        assert gen["prefill_pct"] is None

    def test_prefill_pct_mid_prefill_realistic_values(self):
        """Realistic mid-prefill scenario: 25% of prompt processed."""
        mgr = _FakeMgr()
        poller = LiveSlotsPoller(mgr)

        slots_data = [{
            "is_processing": True,
            "id_task": 1,
            "params": {"max_tokens": 4096, "n_predict": 4096, "stream": True},
            "next_token": [{"n_decoded": 0, "n_remain": 4096, "has_next_token": True}],
            "n_prompt_tokens": 10000,   # running count = 10000
            "n_prompt_tokens_processed": 7500,
            "n_prompt_tokens_cache": 2500,
            "n_ctx": 32768,
        }]

        gen = poller._compute(
            slots_data, 0.0, 123, 1, "s-1", admission_ctx_len=40000
        )
        # OLD: (7500+2500)/10000 = 100%  (broken)
        # NEW: (7500+2500)/40000 = 25%   (correct)
        assert gen["prefill_pct"] == 25, \
            f"Expected 25%, got {gen['prefill_pct']}"
