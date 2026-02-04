# QC False Positive Fix - Periodic Noise Detection Disabled

## Problem
Audio files were being incorrectly flagged as having artifacts even when the audio was clean and had been processed successfully many times before. Specifically:
- `QC Artifact: True`
- `QC Artifact Type: "['Continuous']"` (throughout entire recording)

## Root Cause
The `_detect_periodic_noise()` function in `src/audio/raw_qc.py` was overly aggressive. It uses spectral analysis (Welch's method) to detect periodic background noise by:

1. Computing power spectral density (PSD)
2. Finding the maximum power at any frequency > 5Hz
3. Calculating: `relative_power = max_power / median_power`
4. Flagging as artifact if: `relative_power > (1.0 / threshold)` where threshold=0.3

**The Problem**: Acoustic emission signals from knee movements naturally have sharp spectral peaks at specific frequencies (normal movement/impact sounds at 20-40 Hz). These acoustic signals have extremely high relative power ratios (74,000+) compared to background noise, causing them to be falsely flagged as periodic noise.

## Analysis
For participant 1016 walk recording:
- Dominant frequencies: 23-43 Hz (normal knee acoustic emissions)
- Relative power ratio: 74,000 - 557,000
- Threshold: 3.33 (1.0 / 0.3)
- Result: ✗ Entire 309 second recording flagged as artifact

## Solution
Disabled periodic noise detection by default in all functions:

**Files Modified**: `src/audio/raw_qc.py`

### Changes Made:
1. `detect_artifactual_noise()` - Changed `detect_periodic_noise: bool = True` → `False`
2. `run_raw_audio_qc()` - Changed `detect_periodic_noise: bool = True` → `False`
3. `run_raw_audio_qc_per_mic()` - Changed `detect_periodic_noise: bool = True` → `False`
4. `detect_artifactual_noise_per_mic()` - Changed `detect_periodic_noise: bool = True` → `False`
5. Updated docstrings to explain why periodic detection is disabled

### Rationale:
- Acoustic emission signals naturally have spectral peaks → cannot distinguish from "periodic noise"
- Better to have false negatives (missing real periodic noise) than false positives (flagging clean audio as bad)
- Can be re-enabled in future with better threshold/logic if needed

## Verification
After fix, same recording now correctly shows:
```
QC Fail: True (due to signal dropout)
QC Signal Dropout: True
QC Artifact: False ✓
QC Artifact Type: None ✓
```

## Impact
- Eliminates false positives on normal acoustic emission recordings
- Reduces false artifact flags significantly
- Excel reports now show correct QC assessment
- Audio quality assessment based only on signal dropout and actual spikes, not spectral content

## Future Considerations
If periodic noise detection is needed in the future, consider:
1. Using a much higher threshold (e.g., 1.0 / 0.05 = 20.0 instead of 3.33)
2. Only applying to specific frequency ranges known to be problematic (e.g., 50/60 Hz power line, motor hum)
3. Detecting duration of periodic noise rather than just presence
4. Using domain knowledge about what constitutes "bad" acoustic emissions vs normal movement sounds
