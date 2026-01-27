"""
Object-Oriented Participant Processing Refactoring

This module implements a cleaner, more maintainable approach to processing participant
directories using object-oriented design patterns.

STRUCTURE:
==========

Data Classes (State Encapsulation):
  - AudioData: Holds audio file path, DataFrame, metadata, QC info, and record
  - BiomechanicsData: Holds biomechanics file, recordings, and record
  - SyncData: Holds synchronized audio path, DataFrame, stomp times, and record
  - CycleData: Holds cycle QC results and output paths

Processor Classes (Processing Logic):
  1. ManeuverProcessor: Processes a single maneuver (walk/sit-to-stand/flexion-extension)
     - Inputs: maneuver_dir, maneuver_key, knee_side, study_id, biomechanics_file, biomechanics_type
     - Methods:
       * process_bin_stage() - Load and process raw audio
       * process_sync_stage() - Synchronize audio with biomechanics
       * process_cycles_stage() - Run movement cycle QC
       * save_logs() - Update and save processing logs
     - State: audio, biomechanics, synced_data, cycle_data, log

  2. KneeProcessor: Orchestrates all maneuvers for a single knee
     - Inputs: knee_dir, knee_side, study_id, biomechanics_file, biomechanics_type
     - Methods:
       * process(entrypoint, maneuver) - Process maneuvers from specified stage
       * _find_maneuver_dir() - Find maneuver directories
       * _save_knee_log() - Save knee-level aggregated log
     - State: maneuver_processors, knee_log

  3. ParticipantProcessor: Top-level orchestrator for a complete participant
     - Inputs: participant_dir, biomechanics_type
     - Methods:
       * process(entrypoint, knee, maneuver) - Process participant
       * _find_biomechanics_file() - Find biomechanics Excel file
     - State: knee_processors, study_id, biomechanics_file

ADVANTAGES OVER PREVIOUS DESIGN:
================================

1. Clear State Management:
   - Each object has explicit attributes representing processing state
   - State persists throughout processing, no need to re-fetch or re-calculate
   - Easy to inspect and debug state at any point

2. Single Responsibility:
   - ManeuverProcessor: Processes one maneuver
   - KneeProcessor: Orchestrates maneuvers for one knee
   - ParticipantProcessor: Manages the overall flow
   - Data classes: Hold state without logic

3. No Repeated Method Calls:
   - Instead of calling methods repeatedly and passing results through parameters,
     state is saved in instance attributes
   - Eliminates bugs from passing wrong parameters or losing intermediate results

4. Better Error Handling:
   - Each method can check state before proceeding
   - Failures are logged in context with full state information
   - Easy to resume from specific points

5. Testability:
   - Each class can be tested independently
   - State can be inspected and verified at each stage
   - Fixtures can set up complex test scenarios easily

6. Maintainability:
   - Code is organized by responsibility, not by stage
   - Adding new processing steps is straightforward
   - Reducing parameters and interdependencies makes code easier to understand

EXAMPLE USAGE:
==============

from src.orchestration.participant_processor import ParticipantProcessor

# Create processor for a participant
participant = ParticipantProcessor(
    participant_dir=Path("/path/to/#1011"),
    biomechanics_type="IMU"
)

# Process from bin stage onwards for both knees, all maneuvers
success = participant.process(entrypoint="bin")

# Or process specific knee and maneuver from sync stage
success = participant.process(
    entrypoint="sync",
    knee="Left",
    maneuver="walk"
)

# The processor's state can be inspected:
for knee_side, knee_proc in participant.knee_processors.items():
    for maneuver, man_proc in knee_proc.maneuver_processors.items():
        if man_proc.audio:
            print(f"{knee_side} {maneuver}: {man_proc.audio.df.shape}")
        if man_proc.synced_data:
            print(f"  Synced recordings: {len(man_proc.synced_data)}")

MIGRATION PATH:
===============

The OOP classes are now available and tested. To use them:

1. Update cli/process_directory.py to use ParticipantProcessor instead of process_participant()
2. Remove the long function-based methods from src/orchestration/participant.py
3. Keep the existing low-level functions (create_*_record_from_data, etc.)

The old code should continue to work for now to allow gradual migration.
"""
