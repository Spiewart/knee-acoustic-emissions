# Energy Ratio Validation Maintenance Guide

## What to Tune
- `min_energy_ratio` (default 1.2): recorded knee must be ≥20% louder.
- Δt tolerances: ±0.20s then ±0.30s for peak-pair search.
- RMS window/stride (100 ms, 75% overlap equivalent).

## How to Test
- `pytest tests/test_energy_ratio_validation.py`
- `pytest tests/test_sync_audio_with_biomechanics.py::test_get_audio_stomp_time_dual_knee_selection`
- Optional filters: `pytest -k "energy_ratio or dual_knee"`

## Debug Steps
1) Check detection_results for `audio_stomp_method`, `selected_time`, `contra_selected_time`, `energy_ratio`.
2) Confirm RMS peaks and Δt pairing via debug logs.
3) If ratio fails, consensus is used—verify consensus_time matches log/PNG.

## Common Issues
- Misplaced mic → ratio < 1.2 → falls back to consensus.
- Very close stomps → widen tolerance cautiously.
- Low SNR → consider filtered channels (`f_ch*`) first.
