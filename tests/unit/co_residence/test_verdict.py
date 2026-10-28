"""co_residence.verdict: the re-check fit verdict (no cap anywhere); the arrival verdict has its own file."""

import logging

import pytest
from _co_residence_fakes import FakeHost, FakeResident

from turbohaul.co_residence import DeclineReason, Verdict, fit_verdict

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


# -- fit_verdict: the re-check verdict ------------------------------------------------

async def test_fit_admits_when_a_card_is_picked_and_vram_admits(caplog):
    host = FakeHost(residents=live(1))
    host.pick_result = (2, "none")
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert v == Verdict(True, 2, "none", None, 4000, None), why(f"admit verdict wrong: {v!r}")
    assert lines(caplog) == [], why("an admit writes no decline line")


async def test_fit_asks_the_four_questions_in_order():
    host = FakeHost(residents=live(1))
    await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    asked = [c for c in host.calls if c in {"_route_cap_and_placement", "_resolve_placement_locked",
                                            "_pick_card_for_new_instance", "_vram_admits_locked"}]
    assert asked == ["_route_cap_and_placement", "_resolve_placement_locked",
                     "_pick_card_for_new_instance", "_vram_admits_locked"], why(f"order was {asked!r}")
    assert host.args.get("_vram_admits_locked", []) == [(4000, 1, 1, "none")], why("gate gets need, parallel, card")
    assert host.args.get("_pick_card_for_new_instance", []) == [("m", 4000)], why("picker gets the placement's need")


async def test_fit_declines_no_card_when_the_tag_cannot_be_a_second_engine(caplog):
    host = FakeHost(residents=live(1))
    host.wants_one_card = False
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason) == (False, DeclineReason.NO_CARD), why(f"got {v!r}")
    assert "_resolve_placement_locked" not in host.calls and "_vram_admits_locked" not in host.calls, why(
        "nothing past the placement question may run")
    assert len(lines(caplog)) == 1 and "reason=no_card" in lines(caplog)[0]
    assert "_card_avail_mib" not in host.calls, why("a tag that cannot be a second engine never reads the cards")


async def test_fit_declines_vram_refused_when_the_picker_finds_no_card_but_cards_are_visible(caplog):
    # The picker keeps a safety floor free that the gate does not, so a plain shortage of VRAM
    # shows up as "no card". Cards that can be seen but have no room for one more engine are a
    # VRAM refusal, not a missing card.
    host = FakeHost(residents=live(1))
    host.pick_result = (None, "none")
    host.card_avail = [300, 100]
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason, v.card, v.need) == (False, DeclineReason.VRAM_REFUSED, None, 4000), why(f"got {v!r}")
    assert "_vram_admits_locked" not in host.calls, why("the VRAM gate must not run without a card")
    assert host.calls.count("_card_avail_mib") == 1, why("the visible cards are read once")
    ln = lines(caplog)
    assert len(ln) == 1 and "reason=vram_refused" in ln[0] and "need_mib=4000" in ln[0] and "card=-" in ln[0], why(
        f"decline line was {ln!r}")


async def test_fit_declines_no_card_when_no_card_is_visible(caplog):
    host = FakeHost(residents=live(1))
    host.pick_result = (None, "none")
    host.card_avail = []  # nothing visible, or the probe was unreadable
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason, v.card, v.need) == (False, DeclineReason.NO_CARD, None, 4000), why(f"got {v!r}")
    assert "_vram_admits_locked" not in host.calls and host.calls.count("_card_avail_mib") == 1
    assert len(lines(caplog)) == 1 and "reason=no_card" in lines(caplog)[0]


async def test_fit_reads_the_visible_cards_only_when_the_picker_found_none():
    host = FakeHost(residents=live(1))
    await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    host.vram_ok = False
    await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert "_card_avail_mib" not in host.calls, why("an admit and a gate refusal need no card reading")


async def test_fit_no_card_path_never_reads_the_box_budget():
    host = FakeHost(residents=live(5))
    host.cap_raises = True
    host.pick_result = (None, "none")
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason) == (False, DeclineReason.VRAM_REFUSED), why(f"got {v!r}")
    host.card_avail = []
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason) == (False, DeclineReason.NO_CARD)


async def test_fit_declines_vram_refused_when_the_gate_says_no(caplog):
    host = FakeHost(residents=live(1))
    host.pick_result = (1, "none")
    host.vram_ok = False
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason, v.card, v.split, v.need) == (False, DeclineReason.VRAM_REFUSED, 1, "none", 4000)
    assert "_card_avail_mib" not in host.calls, why("a refused gate on a chosen card needs no card reading")
    ln = lines(caplog)
    assert len(ln) == 1 and "reason=vram_refused" in ln[0] and "card=1" in ln[0], why(f"decline line was {ln!r}")


@pytest.mark.parametrize("eff", [0, 1, 2, 9])
async def test_fit_ignores_the_cap_derived_engine_count(eff):
    host = FakeHost(residents=live(3))
    host.eff_cap = eff
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert v.admit is True, why(f"eff_cap {eff} from the host must not limit the re-check")
    assert v.eff_cap is None, why("the re-check verdict carries no cap")


async def test_fit_never_reads_the_box_budget_even_when_it_would_refuse():
    host = FakeHost(residents=live(5))
    host.cap_raises = True
    ok = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert ok.admit is True
    host.vram_ok = False
    no = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (no.admit, no.reason) == (False, DeclineReason.VRAM_REFUSED), why("a decline path must not read the budget")
    assert "runtime.queue.max_parallel_sidecars" not in host.calls


async def test_fit_admits_far_past_the_old_count_limit():
    host = FakeHost(residents=live(8))
    host.cap = 1
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert v.admit is True, why("eight live engines on a budget of one: only the fit decides")


async def test_fit_decline_line_has_the_recheck_shape(caplog):
    host = FakeHost(residents=live(2))
    host.vram_ok = False
    await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert lines(caplog) == [
        "CO_RESIDENCE_DECLINED via=recheck tag=m reason=vram_refused eff_cap=- instances=2 residents=2 "
        "max_sidecars=- need_mib=4000 card=1 split=none"
    ], why(f"line was {lines(caplog)!r}")


async def test_fit_writes_one_line_per_decline_without_throttling(caplog):
    host = FakeHost(residents=live(1))
    host.vram_ok = False
    for _ in range(3):
        await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert len(lines(caplog)) == 3, why("the re-check is bounded by the memo, so every decline is logged")
    assert len(host.state.declog) == 0, why("the re-check must not touch the arrival throttle")


async def test_fit_survives_a_logging_failure():
    class Broken(FakeHost):
        def _model_residents(self):
            raise RuntimeError("log-time failure")

    host = Broken(residents=live(1))
    host.vram_ok = False
    v = await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
    assert (v.admit, v.reason) == (False, DeclineReason.VRAM_REFUSED), why("a log failure must not change the verdict")


async def test_fit_lets_a_placement_error_through():
    class Broken(FakeHost):
        def _vram_admits_locked(self, *a):
            raise RuntimeError("probe failed")

    host = Broken(residents=live(1))
    with pytest.raises(RuntimeError, match="probe failed"):
        await fit_verdict(host, host.state, "m", host._instances_for("m"), 1000.0)
