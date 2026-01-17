# Energy Ratio Validation for Biomech-Guided Stomp Detection

## Problem
Biomechanics-guided peak pairing could accept contralateral-dominant peaks, leading to wrong stomp selection.

## Solution
- Enforce energy ratio ≥ 1.2 (recorded knee vs contralateral) when selecting peak pairs separated by biomechanics Δt.
- Fall back to consensus when ratio fails.
- Surface `audio_stomp_method`, `selected_time`, `contra_selected_time`, and `energy_ratio` in detection results, logs, and figures.

## Affected Files
- `src/synchronization/sync.py` (energy ratio validation, metadata propagation, plotting)
- `src/orchestration/processing_log.py` (log fields)
- `tests/*` (energy-ratio coverage)

## Tests
- `pytest tests/test_energy_ratio_validation.py`
- `pytest tests/test_sync_audio_with_biomechanics.py::test_get_audio_stomp_time_dual_knee_selection`
- Full suite: `pytest`