# Consensus Stomp Detection: Clustering Method

## Problem
Blind median of 3 detection methods introduces sampling bias—single outlier skews result, poor scientific rigor.

## Solution
**Clustering-based consensus** (±0.5s tolerance):
- Groups methods that agree within ±0.5s of RMS time (primary)
- If >1 method in cluster: uses **mean** of their times (more robust)
- If only 1 method (RMS isolated): uses RMS value alone
- Avoids outlier contamination; ensures only agreeing methods contribute

## Detection Results
Detection results dict now includes:
- `consensus_methods`: List of methods that contributed ('rms', 'onset', 'freq')
- Used in logs and figures to show which methods reached consensus

## Excel Log Enhancements
Two new columns added to Synchronization sheet:
- **"Consensus Methods"**: Methods used for consensus (e.g., "rms, onset, freq" or "rms")
- **"Method Agreement Span (s)"**: Difference between max and min times of methods in consensus cluster
  - Shows tightness of agreement between contributing methods
  - Only methods in the consensus cluster contribute to span (not all three methods)

## Affected Files
- `src/synchronization/sync.py` (consensus clustering logic, detection results, plotting labels)
- `src/orchestration/processing_log.py` (Excel log: new columns, unpacking logic)
- Docstrings updated to document clustering algorithm

## Tests
- `pytest tests/test_energy_ratio_validation.py` (9 tests, all pass)
- `pytest tests/test_sync_audio_with_biomechanics.py` (15 tests, all pass)
- `pytest tests/test_detection_results_fields.py` (all pass)
- `pytest tests/test_processing_log.py` (27 tests, all pass)
- `pytest tests/test_process_participant_directory.py` (59 tests, all pass)

## Tuning
- Tolerance: ±0.5s (hardcoded in `get_audio_stomp_time`)
- Change in lines ~500-530 of sync.py if adjustment needed
