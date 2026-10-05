"""The spawn safety gate refused models on LOAD AVERAGE
while the host sat mostly idle. Load average counts tasks that are
runnable OR blocked (e.g. on disk I/O); it is not CPU utilisation, and it
read high under ambient, unrelated host load with no relationship to spare
CPU. Example capture taken under
ambient, unrelated host load:
    1min-load-per-core=1.072 (gate threshold 0.90, old gate: REFUSE)
    actual CPU busy=36.0% (0.4s /proc/stat delta, same instant)

check_load_avg is replaced in the gate list by check_cpu_util, which reads
real host-wide CPU utilisation from a short /proc/stat delta (same shape as
the existing check_iowait -- two independent samples, not a shared one).
Host-wide scope
is UNCHANGED and deliberate: /proc/stat is not PID-namespaced, so turbohaul
already sees the whole machine from inside its own container -- spawning a
large model loads the whole host, so the gate should see the whole host. Only
the metric changed.

★ TestLoadVsCpuDisagree is THE WHOLE TEST for this change. A fixture that raises
BOTH load average and real CPU proves nothing -- the defect IS that they can
disagree. This fixture forces them to disagree (high load-per-core, low real
CPU, the exact shape above) and asserts the production
entrypoint (all_safety_gates) no longer refuses on it. Every test carries a
POSITIVE HALF proving the disagreement was genuinely reached (old metric
really does refuse under this fixture) -- so a RED here can only mean "the
gate still uses the wrong metric," never "the fixture never got there."
"""
from unittest.mock import patch

from turbohaul.safety import (
    GateResult,
    all_safety_gates,
    check_cpu_util,
    check_load_avg,
)


def _result(results: list[GateResult], name: str) -> GateResult:
    for r in results:
        if r.name == name:
            return r
    raise AssertionError(
        f"no GateResult named {name!r} in {[r.name for r in results]!r}"
    )


class TestCheckCpuUtil:
    """Unit-level correctness of the new gate function itself -- same shape
    as TestCheckIowait in test_safety.py (2-sample /proc/stat delta)."""

    def test_passes_when_busy_low(self):
        # total delta 400, idle delta 380 -> busy 5%
        samples = [(1_000_000, 640_000), (1_000_400, 640_380)]
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies", side_effect=samples,
        ):
            r = check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        assert r.ok
        assert r.name == "cpu_util"
        assert "5.0%" in r.detail

    def test_fails_when_busy_high(self):
        # total delta 400, idle delta 40 -> busy 90%
        samples = [(1_000_000, 640_000), (1_000_400, 640_040)]
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies", side_effect=samples,
        ):
            r = check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        assert not r.ok
        assert "90.0%" in r.detail
        assert "max 85.0%" in r.detail

    def test_passes_no_probe_when_proc_stat_missing(self):
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies", return_value=None,
        ):
            r = check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        assert r.ok
        assert r.detail == "passed-no-probe"

    def test_passes_no_probe_second_when_second_read_fails(self):
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies",
            side_effect=[(1_000_000, 640_000), None],
        ):
            r = check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        assert r.ok
        assert r.detail == "passed-no-probe-second"

    def test_passes_zero_delta(self):
        # A stuck/duplicate read (total didn't advance) must not divide by
        # zero or false-refuse.
        same = (1_000_000, 640_000)
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies",
            side_effect=[same, same],
        ):
            r = check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        assert r.ok
        assert r.detail == "passed-zero-delta"

    def test_host_wide_not_container_scoped_by_construction(self):
        """Host-wide scope is DELIBERATE, not a bug to fix. This
        reads whatever _read_stat_cpu_jiffies (== /proc/stat) reports, with
        no cgroup/PID-namespace filtering anywhere in check_cpu_util -- the
        same unscoped read check_iowait already performs today. A future
        change that container-scopes this function would still leave this
        test's mock in place and green, so this test alone cannot catch that
        regression; it documents the invariant, it does not enforce it
        structurally (enforcing it structurally would require asserting on
        check_cpu_util's source text, which is out of scope for this change)."""
        samples = [(1_000_000, 640_000), (1_000_400, 640_144)]
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies", side_effect=samples,
        ) as mock_read:
            check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        # No cgroup path, no container filter argument was ever passed in.
        for call in mock_read.call_args_list:
            assert call.args == () and call.kwargs == {}


class TestLoadVsCpuDisagree:
    """The fixture where load average and real CPU DISAGREE -- the
    exact shape of the example capture above:
    load-per-core=1.072 (old gate: REFUSE, threshold 0.90),
    actual CPU busy=36.0% over the same instant."""

    _LOAD1, _LOAD5, _LOAD15 = 25.727, 25.727, 22.0
    _CPUS = 24
    # total delta 400, idle delta 256 -> busy 36% (matches the example capture)
    _CPU_SAMPLES = [(1_000_000, 640_000), (1_000_400, 640_256)]

    def test_old_metric_wrongly_refuses_this_exact_disagreement(self):
        """POSITIVE HALF / precondition control: the OLD function, given the
        real captured numbers, really does refuse. If this assertion ever
        fails, the fixture stopped being a genuine disagreement and the rest
        of this class proves nothing -- check_load_avg is unchanged by this
        change (kept, unused, see safety.py), so this must hold on every tree."""
        with patch("os.getloadavg", return_value=(self._LOAD1, self._LOAD5, self._LOAD15)):
            with patch("os.cpu_count", return_value=self._CPUS):
                old = check_load_avg(max_per_core=0.90)
        assert not old.ok, (
            "fixture precondition failed: check_load_avg must refuse under "
            f"this load ({old.detail}) or this is not a real disagreement")

    def test_new_metric_does_not_refuse_the_same_disagreement(self):
        """THE GATE. Real CPU is 36% busy (well under any sane threshold) at
        the exact same instant the old metric refuses. check_cpu_util must
        not be fooled by the ambient load average."""
        with patch(
            "turbohaul.safety._read_stat_cpu_jiffies",
            side_effect=self._CPU_SAMPLES,
        ):
            new = check_cpu_util(max_percent=85.0, sample_window_s=0.01)
        assert new.ok, (
            f"check_cpu_util refused a host that is only 36% busy ({new.detail}) "
            "-- it must not inherit check_load_avg's false-refusal under "
            "ambient, unrelated host load")
        assert "36.0%" in new.detail

    def test_production_gate_list_no_longer_refuses_on_this_disagreement(self):
        """THE REAL CONTROL: all_safety_gates is what manager.py actually
        calls (both call sites). Drive it exactly like production does, under
        the disagreement fixture, and check the cpu_util GateResult by name
        -- not by assuming which function backs it. A revert of the
        safety.py call-site swap (check_cpu_util -> check_load_avg) is this
        test's target: reverting it makes exactly this assertion fail
        (the watched RED arm)."""
        with patch("os.getloadavg", return_value=(self._LOAD1, self._LOAD5, self._LOAD15)):
            with patch("os.cpu_count", return_value=self._CPUS):
                with patch(
                    "turbohaul.safety._read_stat_cpu_jiffies",
                    side_effect=self._CPU_SAMPLES,
                ):
                    results = all_safety_gates(
                        min_free_ram_mib=0,
                        min_free_vram_mib=0,
                        max_cpu_busy_percent=85.0,
                        max_iowait_percent=100.0,
                        cpu_util_sample_window_s=0.01,
                        iowait_sample_window_s=0.0,
                    )
        gate = _result(results, "cpu_util")
        assert gate.ok, (
            f"production gate list refused under ambient load-average noise "
            f"with real CPU at 36% busy: {gate.detail}")
        # NEGATIVE half: the retired name must not silently reappear in the
        # gate list (would mean the call-site swap was only partially done).
        assert not any(r.name == "cpu_load" for r in results), (
            "cpu_load (the old, retired gate) is still present in "
            "all_safety_gates' output -- the call-site swap is incomplete")
