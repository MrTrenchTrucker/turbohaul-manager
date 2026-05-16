"""Tests for safety guardrails (Cmdr #15653)."""
from unittest.mock import patch

from turbohaul.safety import (
    GateResult,
    all_safety_gates,
    check_free_ram,
    check_load_avg,
    check_iowait,
)


class TestCheckFreeRam:
    def test_passes_when_above_threshold(self):
        with patch(
            "turbohaul.safety._read_meminfo_kib",
            return_value={"MemAvailable": 4 * 1024 * 1024},  # 4 GiB
        ):
            r = check_free_ram(min_free_mib=1024)
        assert r.ok
        assert r.name == "ram"

    def test_fails_when_below_threshold(self):
        with patch(
            "turbohaul.safety._read_meminfo_kib",
            return_value={"MemAvailable": 256 * 1024},  # 256 MiB
        ):
            r = check_free_ram(min_free_mib=1024)
        assert not r.ok
        assert "only 256 MiB free" in r.detail

    def test_passes_no_probe_when_meminfo_unavailable(self):
        with patch("turbohaul.safety._read_meminfo_kib", return_value={}):
            r = check_free_ram(min_free_mib=1024)
        assert r.ok
        assert r.detail == "passed-no-probe"


class TestCheckLoadAvg:
    def test_passes_when_load_low(self):
        with patch("os.getloadavg", return_value=(0.5, 0.3, 0.2)):
            with patch("os.cpu_count", return_value=8):
                r = check_load_avg(max_per_core=0.9)
        assert r.ok

    def test_fails_when_load_high(self):
        with patch("os.getloadavg", return_value=(16.0, 12.0, 8.0)):
            with patch("os.cpu_count", return_value=8):
                r = check_load_avg(max_per_core=0.9)
        assert not r.ok
        assert "2.00" in r.detail or "2.0" in r.detail


class TestCheckIowait:
    def test_passes_no_probe_when_proc_stat_missing(self):
        with patch(
            "turbohaul.safety._read_stat_iowait_jiffies", return_value=None,
        ):
            r = check_iowait(max_percent=30.0, sample_window_s=0.01)
        assert r.ok
        assert "passed-no-probe" in r.detail

    def test_passes_when_iowait_low(self):
        samples = [(1000, 10), (1100, 12)]  # 2/100 = 2%
        with patch(
            "turbohaul.safety._read_stat_iowait_jiffies",
            side_effect=samples,
        ):
            r = check_iowait(max_percent=30.0, sample_window_s=0.01)
        assert r.ok
        assert "iowait 2.0%" in r.detail

    def test_fails_when_iowait_high(self):
        samples = [(1000, 100), (1100, 200)]  # delta 100/100 = 100%
        with patch(
            "turbohaul.safety._read_stat_iowait_jiffies",
            side_effect=samples,
        ):
            r = check_iowait(max_percent=30.0, sample_window_s=0.01)
        assert not r.ok
        assert "100.0%" in r.detail


class TestAllGatesAggregate:
    def test_returns_4_gates(self):
        with patch("turbohaul.safety._read_meminfo_kib", return_value={}):
            with patch(
                "turbohaul.safety._read_free_vram_mib", return_value=None,
            ):
                with patch(
                    "turbohaul.safety._read_stat_iowait_jiffies",
                    return_value=None,
                ):
                    with patch("os.getloadavg", return_value=(0.1, 0.1, 0.1)):
                        with patch("os.cpu_count", return_value=4):
                            results = all_safety_gates(
                                min_free_ram_mib=1024,
                                min_free_vram_mib=512,
                                max_load_per_core=0.9,
                                max_iowait_percent=30.0,
                                iowait_sample_window_s=0.01,
                            )
        assert len(results) == 4
        names = {g.name for g in results}
        assert names == {"ram", "vram", "cpu_load", "iowait"}
