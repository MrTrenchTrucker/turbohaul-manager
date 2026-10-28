"""co_residence.verdict: the arrival verdict (the arrival-time admission logic, with the decline line)."""

import logging

import pytest
from _co_residence_fakes import FakeHost, FakeResident, FakeSlot

from turbohaul.co_residence import DeclineReason, Verdict, arrival_verdict

LOGGER = "turbohaul.co_residence.verdict"


def why(text: str) -> str:
    """Prefix a failure message with the module name and a pointer to its card."""
    return f"co_residence: {text} (the contract is in src/turbohaul/co_residence/AGENTS.md, section Invariants)"


def live(n: int) -> list[FakeResident]:
    return [FakeResident(resident_key=f"m#{i}") for i in range(n)]


def lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and "CO_RESIDENCE_DECLINED" in r.getMessage()]


@pytest.fixture(autouse=True)
def _info(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)


# -- arrival_verdict: the arrival-time admission logic ----------------------------------------------

async def test_arrival_admits_below_both_limits():
    host = FakeHost(residents=live(1))
    host.cap = 3
    v = await arrival_verdict(host, "m", host._instances_for("m"), 2, FakeSlot())
    assert v == Verdict(True, 1, "none", None, 4000, 2), why(f"admit verdict wrong: {v!r}")


@pytest.mark.parametrize(
    "insts, eff_cap, residents, budget, expected",
    [
        (1, 2, 2, 3, True),                       # room on both counts
        (2, 2, 2, 3, DeclineReason.CAP),          # the tag is at its own ceiling
        (1, 2, 3, 3, DeclineReason.BOX_FULL),     # the box budget is spent
        (3, 2, 3, 3, DeclineReason.CAP),          # both: the tag ceiling is named first
        (1, 0, 1, 3, DeclineReason.CAP),          # a zero ceiling declines
        (1, 2, 4, 3, DeclineReason.BOX_FULL),     # over budget counts as full
    ],
)
async def test_arrival_cap_condition_both_ways(insts, eff_cap, residents, budget, expected):
    mine = live(insts)
    others = [FakeResident(model_tag="o", resident_key=f"o{i}") for i in range(residents - insts)]
    host = FakeHost(residents=mine + others)
    host.cap = budget
    v = await arrival_verdict(host, "m", host._instances_for("m"), eff_cap, FakeSlot())
    if expected is True:
        assert v.admit is True, why(f"{insts}/{eff_cap}/{residents}/{budget} should admit, got {v!r}")
    else:
        assert (v.admit, v.reason) == (False, expected), why(f"{insts}/{eff_cap}/{residents}/{budget} gave {v!r}")
        assert "_resolve_placement_locked" not in host.calls, why("a capped arrival must not probe the cards")
    assert v.eff_cap == eff_cap


async def test_arrival_declines_no_card_and_vram_refused():
    host = FakeHost(residents=live(1))
    host.pick_result = (None, "none")
    v = await arrival_verdict(host, "m", host._instances_for("m"), 2, FakeSlot())
    assert (v.admit, v.reason) == (False, DeclineReason.NO_CARD), why("the arrival path keeps no_card for a None pick")
    assert "_card_avail_mib" not in host.calls, why("the arrival path does not read the visible cards")
    host2 = FakeHost(residents=live(1))
    host2.vram_ok = False
    v2 = await arrival_verdict(host2, "m", host2._instances_for("m"), 2, FakeSlot())
    assert (v2.admit, v2.reason, v2.card) == (False, DeclineReason.VRAM_REFUSED, 1)


async def test_arrival_fast_lane_relocation_decides_a_cold_start_card():
    host = FakeHost(residents=[])
    slot = FakeSlot(fastlane_relocated_main_gpu=3, fastlane_relocated_split_mode="none")
    v = await arrival_verdict(host, "m", [], 1, slot)
    assert (v.admit, v.card, v.split) == (True, 3, "none"), why(f"relocation ignored: {v!r}")
    assert host.args.get("_vram_admits_locked", [[]])[0][2:] == (3, "none"), why("gate tests the relocated card")


async def test_arrival_fast_lane_relocation_never_overrides_a_second_copy():
    host = FakeHost(residents=live(1))
    slot = FakeSlot(fastlane_relocated_main_gpu=3, fastlane_relocated_split_mode="none")
    v = await arrival_verdict(host, "m", host._instances_for("m"), 2, slot)
    assert v.card == 1, why("a tag that already has an engine keeps the picker's card")


async def test_arrival_relocation_with_a_card_none_pick_still_admits_on_the_relocated_card():
    host = FakeHost(residents=[])
    host.pick_result = (None, "none")
    slot = FakeSlot(fastlane_relocated_main_gpu=2, fastlane_relocated_split_mode="none")
    v = await arrival_verdict(host, "m", [], 1, slot)
    assert (v.admit, v.card) == (True, 2), why("the relocation replaces the picker's result, even a None")


async def test_arrival_without_a_slot_has_no_relocation():
    host = FakeHost(residents=live(1))
    v = await arrival_verdict(host, "m", host._instances_for("m"), 2, None)
    assert v.admit is True and v.card == 1


async def test_arrival_decline_line_is_written_for_a_tag_that_already_runs(caplog):
    host = FakeHost(residents=live(1))
    host.cap = 1
    await arrival_verdict(host, "m", host._instances_for("m"), 1, FakeSlot())
    assert lines(caplog) == [
        "CO_RESIDENCE_DECLINED via=arrival tag=m reason=cap eff_cap=1 instances=1 residents=1 "
        "max_sidecars=1 need_mib=- card=- split=-"
    ], why(f"line was {lines(caplog)!r}")


async def test_arrival_decline_line_is_written_once_per_window_per_reason(caplog):
    host = FakeHost(residents=live(1))
    host.cap = 1
    for _ in range(3):
        await arrival_verdict(host, "m", host._instances_for("m"), 1, FakeSlot())
    assert len(lines(caplog)) == 1, why("the same verdict repeats on every arrival: log it once per window")
    host.now += 59.0
    await arrival_verdict(host, "m", host._instances_for("m"), 1, FakeSlot())
    assert len(lines(caplog)) == 1, why("still inside the 60 s window")
    host.now += 1.0
    await arrival_verdict(host, "m", host._instances_for("m"), 1, FakeSlot())
    assert len(lines(caplog)) == 2, why("a new window logs again")
    host.cap = 9
    host.vram_ok = False
    await arrival_verdict(host, "m", host._instances_for("m"), 2, FakeSlot())
    assert len(lines(caplog)) == 3 and "reason=vram_refused" in lines(caplog)[-1], why("each reason has its own key")


async def test_arrival_throttle_is_per_tag():
    host = FakeHost(residents=live(1) + [FakeResident(model_tag="n", resident_key="n")])
    host.cap = 2
    await arrival_verdict(host, "m", host._instances_for("m"), 1, FakeSlot())
    await arrival_verdict(host, "n", host._instances_for("n"), 1, FakeSlot(model_tag="n"))
    assert host.state.declog.keys() == ["m|cap", "n|cap"], why(f"throttle keys were {host.state.declog.keys()!r}")


async def test_arrival_writes_no_line_for_a_cold_tag(caplog):
    host = FakeHost(residents=[])
    host.vram_ok = False
    v = await arrival_verdict(host, "m", [], 1, FakeSlot())
    assert (v.admit, v.reason) == (False, DeclineReason.VRAM_REFUSED)
    host.cap = 0
    await arrival_verdict(host, "m", [], 1, FakeSlot())
    assert lines(caplog) == [], why("a tag with no engine is not a co-residence question")
    assert len(host.state.declog) == 0


async def test_arrival_survives_a_logging_failure():
    class Broken(FakeHost):
        def _co_residence_now(self):
            raise RuntimeError("clock failed")

    host = Broken(residents=live(1))
    host.cap = 1
    v = await arrival_verdict(host, "m", host._instances_for("m"), 1, FakeSlot())
    assert (v.admit, v.reason) == (False, DeclineReason.CAP), why("a log failure must not change the verdict")


async def test_arrival_reads_the_box_budget():
    host = FakeHost(residents=live(1))
    await arrival_verdict(host, "m", host._instances_for("m"), 2, FakeSlot())
    assert "runtime.queue.max_parallel_sidecars" in host.calls, why("the arrival path is the capped one")
