"""Contract test for the engine_budget module's public interface.

Asserts that engine_budget exports exactly the registry's `public` list
(modules.toml [modules.engine_budget]) and that each name is usable.
"""
import turbohaul.engine_budget as engine_budget

EXPECTED_PUBLIC = {"EngineBudgetInputs", "EngineBudget", "effective_engine_cap", "context_window_capacity"}


def test_exports_exactly_the_registry_public_names():
    assert set(engine_budget.__all__) == EXPECTED_PUBLIC


def test_every_public_name_is_present_and_callable():
    for name in EXPECTED_PUBLIC:
        assert hasattr(engine_budget, name), f"engine_budget has no attribute {name!r}"
        assert callable(getattr(engine_budget, name)), f"engine_budget.{name} is not callable"
