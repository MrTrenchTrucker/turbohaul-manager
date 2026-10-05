# Decisions

This folder holds the decisions that shaped the code: why it is the way it is, one decision per file. Each file is an ADR (architecture decision record): a short note of a decision, why it was made, and what it changed. Every file has the same shape: a Status and a Date, then Context, Decision, Reasons and Consequences.

- [ADR-001](ADR-001-module-system-and-engine-launch.md): Adopt the module system one fix at a time; first module engine_launch (accepted, 2026-09-25)
- [ADR-002](ADR-002-retire-max-instances-engine-budget.md): Retire the per-model max_instances setting; the box-wide sidecar budget decides (module engine_budget) (accepted, 2026-10-01)
- [ADR-003](ADR-003-discovered-clients-show-container-names.md): The Fast Lane Discovered list shows a confirmed container name, and adding from it saves a rule by name (module fastlane_client_names) (accepted, 2026-10-01)
