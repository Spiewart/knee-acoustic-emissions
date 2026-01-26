#!/usr/bin/env python3
"""
Script to fix test_metadata_validators.py by updating model instantiations
to match the new required fields and type expectations.

Issues to fix:
1. firmware_version: change from string "1.0.0" to int (e.g., 1)
2. recording_date: add datetime(2024, 1, 1)
3. knee: add "left" or appropriate value
4. maneuver: add "walk" or appropriate value (for AcousticsFile-based classes)
5. Handle timedelta vs datetime mismatches
"""

import re
from pathlib import Path
from typing import List, Tuple

# Define the test file path
TEST_FILE = Path("/Users/spiewart/acoustic_emissions_processing/tests/test_metadata_validators.py")

# Read the test file
with open(TEST_FILE, 'r') as f:
    content = f.read()

# List of all replacements needed
replacements: List[Tuple[str, str, str]] = []

# ============================================================================
# PATTERN 1: Fix firmware_version from string to int
# ============================================================================

# Pattern: firmware_version="1.0.0" -> firmware_version=1
pattern1_matches = re.finditer(r'firmware_version="1\.0\.0"', content)
for match in pattern1_matches:
    old = match.group(0)
    new = 'firmware_version=1'
    replacements.append((old, new, "Fix firmware_version type from string to int"))

# ============================================================================
# PATTERN 2: Add recording_date where missing in AcousticsFile instances
# ============================================================================

# Look for AcousticsFile instantiations that have recording_time but no recording_date
acoustics_file_pattern = r'AcousticsFile\(([\s\S]*?)\)'
for match in re.finditer(acoustics_file_pattern, content):
    block = match.group(0)
    block_start = match.start()

    # Check if recording_time exists but recording_date doesn't
    if 'recording_time=' in block and 'recording_date=' not in block:
        # Find where to insert recording_date (before recording_time)
        insert_pos = block.find('recording_time=')
        if insert_pos != -1:
            # Create the fixed version with recording_date added
            old = block
            # Insert recording_date before recording_time
            indent_match = re.search(r'(\s+)recording_time=', block)
            if indent_match:
                indent = indent_match.group(1)
                new = block[:insert_pos] + f'recording_date=datetime(2024, 1, 1),\n{indent}recording_time=' + block[insert_pos + len('recording_time='):]
                replacements.append((old, new, "Add recording_date to AcousticsFile instance"))

# ============================================================================
# PATTERN 3: Add knee and maneuver where missing in AcousticsFile instances
# ============================================================================

for match in re.finditer(acoustics_file_pattern, content):
    block = match.group(0)

    # Check if knee is missing
    if 'knee=' not in block and 'study=' in block:
        # Find position after study_id
        insert_pos = block.find('study_id=')
        if insert_pos != -1:
            # Find the end of the study_id line
            line_end = block.find('\n', insert_pos)
            if line_end != -1:
                old = block
                indent_match = re.search(r'(\s+)study_id=', block)
                if indent_match:
                    indent = indent_match.group(1)
                    new = block[:line_end+1] + f'{indent}knee="left",\n' + block[line_end+1:]
                    if (old, new) not in replacements and old in content:
                        replacements.append((old, new, "Add knee='left' to AcousticsFile instance"))

    # Check if maneuver is missing (for AcousticsFile-based classes)
    if 'maneuver=' not in block and 'knee=' in block and 'AcousticsFile' in block:
        old = block
        # Add maneuver after knee
        insert_pos = block.find('knee=')
        if insert_pos != -1:
            line_end = block.find('\n', insert_pos)
            if line_end != -1:
                indent_match = re.search(r'(\s+)knee=', block)
                if indent_match:
                    indent = indent_match.group(1)
                    new = block[:line_end+1] + f'{indent}maneuver="walk",\n' + block[line_end+1:]
                    if (old, new) not in replacements and old in content:
                        replacements.append((old, new, "Add maneuver='walk' to AcousticsFile instance"))

# ============================================================================
# PATTERN 4: Handle other model classes
# ============================================================================

# SynchronizationMetadata - add missing required fields
sync_meta_pattern = r'SynchronizationMetadata\(([\s\S]*?)\)'
for match in re.finditer(sync_meta_pattern, content):
    block = match.group(0)

    # Check for missing fields
    required_fields = ['audio_sync_time', 'sync_offset', 'aligned_audio_sync_time',
                      'aligned_bio_sync_time', 'sync_method', 'consensus_time',
                      'rms_time', 'onset_time', 'freq_time']

    missing_fields = [f for f in required_fields if f'{f}=' not in block]

    if missing_fields and 'firmware_version=' in block:
        # Need to add the missing fields
        old = block
        # Find a good insertion point (after firmware_version line)
        insert_pos = block.rfind('firmware_version=')
        if insert_pos != -1:
            line_end = block.find('\n', insert_pos)
            if line_end != -1:
                # Get indent from the line
                line_start = block.rfind('\n', 0, insert_pos) + 1
                indent = block[line_start:insert_pos]

                # Build the additions
                additions = []
                if 'sync_offset=' not in block:
                    additions.append(f'{indent}sync_offset=timedelta(seconds=1.5),')
                if 'audio_sync_time=' not in block:
                    additions.append(f'{indent}audio_sync_time=timedelta(seconds=10.0),')
                if 'aligned_audio_sync_time=' not in block:
                    additions.append(f'{indent}aligned_audio_sync_time=timedelta(seconds=10.0),')
                if 'aligned_bio_sync_time=' not in block:
                    additions.append(f'{indent}aligned_bio_sync_time=timedelta(seconds=8.5),')
                if 'sync_method=' not in block:
                    additions.append(f'{indent}sync_method="consensus",')
                if 'consensus_time=' not in block:
                    additions.append(f'{indent}consensus_time=timedelta(seconds=10.0),')
                if 'rms_time=' not in block:
                    additions.append(f'{indent}rms_time=timedelta(seconds=10.0),')
                if 'onset_time=' not in block:
                    additions.append(f'{indent}onset_time=timedelta(seconds=10.0),')
                if 'freq_time=' not in block:
                    additions.append(f'{indent}freq_time=timedelta(seconds=10.0),')

                if additions:
                    new = block[:line_end] + '\n' + '\n'.join(additions) + block[line_end:]
                    if (old, new) not in replacements and old in content:
                        replacements.append((old, new, "Add missing required fields to SynchronizationMetadata"))

# ============================================================================
# AudioProcessing - add missing required fields
# ============================================================================

audio_proc_pattern = r'AudioProcessing\(([\s\S]*?)\)'
for match in re.finditer(audio_proc_pattern, content):
    block = match.group(0)

    # Check for missing required fields in AudioProcessing
    if 'qc_fail_segments=' not in block and 'processing_date=' in block:
        old = block
        # Find insertion point after processing_date
        insert_pos = block.find('processing_date=')
        if insert_pos != -1:
            line_end = block.find('\n', insert_pos)
            if line_end != -1:
                line_start = block.rfind('\n', 0, insert_pos) + 1
                indent = block[line_start:insert_pos]

                additions = [
                    f'{indent}qc_fail_segments=[],',
                    f'{indent}qc_fail_segments_ch1=[],',
                    f'{indent}qc_fail_segments_ch2=[],',
                    f'{indent}qc_fail_segments_ch3=[],',
                    f'{indent}qc_fail_segments_ch4=[],',
                    f'{indent}qc_signal_dropout=False,',
                    f'{indent}qc_signal_dropout_segments=[],',
                    f'{indent}qc_signal_dropout_ch1=False,',
                    f'{indent}qc_signal_dropout_segments_ch1=[],',
                    f'{indent}qc_signal_dropout_ch2=False,',
                    f'{indent}qc_signal_dropout_segments_ch2=[],',
                    f'{indent}qc_signal_dropout_ch3=False,',
                    f'{indent}qc_signal_dropout_segments_ch3=[],',
                    f'{indent}qc_signal_dropout_ch4=False,',
                    f'{indent}qc_signal_dropout_segments_ch4=[],',
                    f'{indent}qc_artifact=False,',
                    f'{indent}qc_artifact_segments=[],',
                    f'{indent}qc_artifact_ch1=False,',
                    f'{indent}qc_artifact_segments_ch1=[],',
                    f'{indent}qc_artifact_ch2=False,',
                    f'{indent}qc_artifact_segments_ch2=[],',
                    f'{indent}qc_artifact_ch3=False,',
                    f'{indent}qc_artifact_segments_ch3=[],',
                    f'{indent}qc_artifact_ch4=False,',
                    f'{indent}qc_artifact_segments_ch4=[],',
                ]

                new = block[:line_end] + '\n' + '\n'.join(additions) + block[line_end:]
                if (old, new) not in replacements and old in content:
                    replacements.append((old, new, "Add missing required fields to AudioProcessing"))

# ============================================================================
# BiomechanicsImport
# ============================================================================

biomech_import_pattern = r'BiomechanicsImport\(([\s\S]*?)\)'
for match in re.finditer(biomech_import_pattern, content):
    block = match.group(0)

    # Check for missing required fields
    required = ['sheet_name', 'processing_date', 'sample_rate', 'num_sub_recordings',
                'num_passes', 'num_data_points', 'duration_seconds']

    if 'study=' in block:
        missing = [f for f in required if f'{f}=' not in block]
        if missing:
            # Build the completion
            old = block
            # Find the last field to determine insertion point
            last_comma = block.rfind(',')
            if last_comma == -1:
                # No trailing comma, find the last field
                last_eq = block.rfind('=')
            else:
                last_eq = block.rfind('=', 0, last_comma)

            if last_eq != -1:
                line_end = block.find('\n', last_eq)
                if line_end == -1:
                    line_end = len(block) - 1  # Before closing paren

                line_start = block.rfind('\n', 0, last_eq) + 1
                indent = block[line_start:last_eq-len(block[last_eq:].split('=')[0])]

                additions = []
                if 'sheet_name=' not in block:
                    additions.append(f'{indent}sheet_name="Sheet1",')
                if 'processing_date=' not in block:
                    additions.append(f'{indent}processing_date=datetime(2024, 1, 1),')
                if 'sample_rate=' not in block:
                    additions.append(f'{indent}sample_rate=200.0,')
                if 'num_sub_recordings=' not in block:
                    additions.append(f'{indent}num_sub_recordings=1,')
                if 'num_passes=' not in block:
                    additions.append(f'{indent}num_passes=1,')
                if 'num_data_points=' not in block:
                    additions.append(f'{indent}num_data_points=5000,')
                if 'duration_seconds=' not in block:
                    additions.append(f'{indent}duration_seconds=25.0,')

                if additions:
                    # Insert before closing paren
                    close_paren_pos = block.rfind(')')
                    new = block[:close_paren_pos] + '\n' + '\n'.join(additions) + '\n' + block[close_paren_pos:]
                    if (old, new) not in replacements and old in content:
                        replacements.append((old, new, "Add missing required fields to BiomechanicsImport"))

# ============================================================================
# Generate Report
# ============================================================================

print("=" * 80)
print("TEST FILE FIX REPORT")
print("=" * 80)
print(f"\nFile: {TEST_FILE}")
print(f"Total replacements to apply: {len(replacements)}\n")

for i, (old_str, new_str, reason) in enumerate(replacements, 1):
    print(f"\n[{i}] {reason}")
    print(f"    Old: {old_str[:80]}..." if len(old_str) > 80 else f"    Old: {old_str}")
    print(f"    New: {new_str[:80]}..." if len(new_str) > 80 else f"    New: {new_str}")

# ============================================================================
# Apply Replacements
# ============================================================================

print("\n" + "=" * 80)
print("APPLYING REPLACEMENTS...")
print("=" * 80)

modified_content = content
applied_count = 0
failed_count = 0

for old_str, new_str, reason in replacements:
    if old_str in modified_content:
        modified_content = modified_content.replace(old_str, new_str, 1)
        applied_count += 1
        print(f"✓ Applied: {reason}")
    else:
        failed_count += 1
        print(f"✗ Failed: {reason} (pattern not found)")

# Write back to file
with open(TEST_FILE, 'w') as f:
    f.write(modified_content)

print("\n" + "=" * 80)
print(f"SUMMARY: {applied_count} replacements applied, {failed_count} failed")
print("=" * 80)

if applied_count > 0:
    print(f"\n✓ Successfully fixed {TEST_FILE}")
    print("\nRun tests with: pytest tests/test_metadata_validators.py")
