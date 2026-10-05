"""TurboHaul OOM-requeue — classify_load_failure().

Proves the 3-way partition: 'oom' only on
an OOM-shaped line in scan_engine_log_for_errors()'s own output; 'unknown'
when that scan could not confirm anything either way; 'other' for every
readable-but-non-OOM shape. Only 'oom' may ever requeue (asserted at the
manager.py wiring layer, not here — this module is pure classification).

The 'oom' fixture is representative crash text from a vision-model load
crash (an allocator failure while loading the model), copied verbatim from a
real engine log so this test is anchored to an actual failure shape, not an
invented one.
"""

from turbohaul import load_verify_log as lv

_REAL_OOM_LINE = (
    "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 884.62 MiB on "
    "device 0: cudaMalloc failed: out of memory"
)


def test_real_incident_oom_line_classifies_as_oom():
    scan = {
        "engine_errors_detected": True,
        "engine_error_lines": [_REAL_OOM_LINE],
        "reason": None,
    }
    assert lv.classify_load_failure(scan) == "oom"


def test_failed_to_allocate_variant_classifies_as_oom():
    scan = {
        "engine_errors_detected": True,
        "engine_error_lines": ["E ggml_gallocr_reserve: failed to allocate CUDA0 buffer"],
        "reason": None,
    }
    assert lv.classify_load_failure(scan) == "oom"


def test_oom_match_is_case_insensitive():
    scan = {
        "engine_errors_detected": True,
        "engine_error_lines": ["E cudaMalloc failed: OUT OF MEMORY"],
        "reason": None,
    }
    assert lv.classify_load_failure(scan) == "oom"


def test_non_oom_engine_error_classifies_as_other():
    # A real, distinct failure shape: a missing GGUF blob, not an allocator error.
    scan = {
        "engine_errors_detected": True,
        "engine_error_lines": ["E load_model: failed to open GGUF file: No such file or directory"],
        "reason": None,
    }
    assert lv.classify_load_failure(scan) == "other"


def test_clean_readable_log_that_still_failed_classifies_as_other():
    # engine_errors_detected=False: the log WAS read and had nothing E-level in
    # it, but the health check still failed (e.g. a non-engine-log cause). This
    # is not OOM and not "unprovable" — it is a readable, negative result.
    scan = {"engine_errors_detected": False, "engine_error_lines": [], "reason": None}
    assert lv.classify_load_failure(scan) == "other"


def test_unreadable_log_classifies_as_unknown():
    scan = {
        "engine_errors_detected": None,
        "engine_error_lines": [],
        "reason": "neither log file was readable",
    }
    assert lv.classify_load_failure(scan) == "unknown"


def test_empty_readable_log_classifies_as_unknown():
    scan = {
        "engine_errors_detected": None,
        "engine_error_lines": [],
        "reason": "log file(s) present but empty (0 lines read)",
    }
    assert lv.classify_load_failure(scan) == "unknown"


def test_malformed_scan_degrades_to_unknown_never_oom():
    # Never-fabricate discipline: garbage input must never resolve to the one
    # verdict that would authorize a requeue.
    assert lv.classify_load_failure({}) == "unknown"  # missing key -> detected=None
    assert lv.classify_load_failure(
        {"engine_errors_detected": True, "engine_error_lines": "not-a-list"}
    ) == "unknown"
    assert lv.classify_load_failure(None) == "unknown"


def test_detected_true_with_no_lines_is_other_not_unknown():
    # detected=True but the (optional) lines key is absent/empty is still a
    # READ, definite-negative result, not an unprovable one.
    assert lv.classify_load_failure({"engine_errors_detected": True}) == "other"
    assert lv.classify_load_failure(
        {"engine_errors_detected": True, "engine_error_lines": []}
    ) == "other"


def test_non_string_lines_are_skipped_not_fatal():
    scan = {
        "engine_errors_detected": True,
        "engine_error_lines": [None, 123, _REAL_OOM_LINE],
        "reason": None,
    }
    assert lv.classify_load_failure(scan) == "oom"
