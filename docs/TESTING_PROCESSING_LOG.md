# Testing Documentation for Processing Log System

## Overview

The processing log system has comprehensive test coverage across three test files:

1. **`test_processing_log.py`** - Unit tests for core functionality
2. **`test_processing_log_integration.py`** - Integration tests with orchestration layer
3. **`test_processing_log_standalone.py`** - Standalone validation script

## Test Coverage

### Unit Tests (`test_processing_log.py`)

**Data Model Tests:**
- ✅ `TestAudioProcessingRecord` - Audio record creation, conversion to dict, default values
- ✅ `TestBiomechanicsImportRecord` - Biomechanics record creation and serialization
- ✅ `TestSynchronizationRecord` - Sync record creation and serialization
- ✅ `TestMovementCyclesRecord` - Cycles record creation and serialization

**Core Functionality Tests:**
- ✅ `TestManeuverProcessingLog` - Log creation, CRUD operations for all record types
- ✅ Excel save/load functionality with data preservation
- ✅ `get_or_create()` behavior (creates new vs. loads existing)
- ✅ Incremental updates (updating existing records vs. adding new ones)

**Helper Function Tests:**
- ✅ `TestHelperFunctions` - Creating records from DataFrames and metadata
- ✅ `create_audio_record_from_data()` - Audio statistics calculation, metadata extraction
- ✅ `create_biomechanics_record_from_data()` - Recording parsing, pass counting
- ✅ `create_sync_record_from_data()` - Sync data extraction from DataFrames
- ✅ `create_cycles_record_from_data()` - Cycle count aggregation

**Incremental Update Tests:**
- ✅ `TestIncrementalUpdates` - Verifying partial updates preserve other data
- ✅ Record replacement (same filename = update, not duplicate)
- ✅ Full roundtrip preservation (save → load → all data intact)

**Total: 30+ unit tests**

### Integration Tests (`test_processing_log_integration.py`)

**Orchestration Integration:**
- ✅ `_save_or_update_processing_log()` creates log files correctly
- ✅ Incremental updates preserve existing data (audio + sync separately)
- ✅ Multiple update cycles (audio → sync → sync update)
- ✅ `get_or_create()` integration with file system

**Real-world Scenarios:**
- ✅ Re-processing same file updates existing record
- ✅ Processing different files adds new records
- ✅ Log persists across multiple processing runs

**Total: 5 integration tests**

### Standalone Validation (`test_processing_log_standalone.py`)

**Quick Validation Script:**
- ✅ Basic CRUD operations
- ✅ Excel save/load roundtrip
- ✅ Helper function validation
- ✅ Error handling (failures, non-existent files)
- ✅ Can run without pytest for quick checks

## Running Tests

### Run All Tests (Pytest)

```bash
# Run all processing log tests
pytest tests/test_processing_log.py tests/test_processing_log_integration.py -v

# Run with coverage
pytest tests/test_processing_log*.py --cov=src.orchestration.processing_log --cov-report=html

# Run specific test class
pytest tests/test_processing_log.py::TestManeuverProcessingLog -v

# Run specific test
pytest tests/test_processing_log.py::TestManeuverProcessingLog::test_save_to_excel -v
```

### Run Standalone Validation

```bash
# Quick validation without pytest
python tests/test_processing_log_standalone.py

# Should output:
# ✅ ALL TESTS PASSED!
```

## Test Structure

### Unit Test Example

```python
def test_save_to_excel(self, tmp_path):
    """Test saving log to Excel file."""
    log = ManeuverProcessingLog(
        study_id="1011",
        knee_side="Left",
        maneuver="walk",
        maneuver_directory=tmp_path,
    )

    # Add data
    log.update_audio_record(AudioProcessingRecord(
        audio_file_name="test_audio",
        processing_status="success",
    ))

    # Save
    excel_path = tmp_path / "test_log.xlsx"
    saved_path = log.save_to_excel(excel_path)

    # Verify
    assert saved_path.exists()
    assert saved_path == excel_path
```

### Integration Test Example

```python
def test_incremental_update_preserves_existing_data(self, tmp_path):
    """Test that updating a log preserves existing data."""
    maneuver_dir = tmp_path / "Left Knee" / "Walking"

    # First update: Audio
    _save_or_update_processing_log(
        study_id="1011",
        knee_side="Left",
        maneuver_key="walk",
        maneuver_dir=maneuver_dir,
        audio_df=audio_dataframe,
    )

    # Second update: Sync (should preserve audio)
    _save_or_update_processing_log(
        study_id="1011",
        knee_side="Left",
        maneuver_key="walk",
        maneuver_dir=maneuver_dir,
        synced_data=sync_data,
    )

    # Verify both present
    loaded_log = ManeuverProcessingLog.load_from_excel(log_path)
    assert loaded_log.audio_record is not None
    assert len(loaded_log.synchronization_records) == 1
```

## Test Data

### Fixtures

Tests use `pytest`'s `tmp_path` fixture for temporary directories, ensuring:
- No pollution of real file system
- Automatic cleanup after tests
- Isolation between tests

### Sample Data Generation

Tests generate realistic sample data:

```python
# Audio DataFrame
audio_df = pd.DataFrame({
    'tt': pd.date_range('2024-01-01', periods=1000, freq='21.333us'),
    'ch1': np.random.randn(1000) * 150,
    'ch2': np.random.randn(1000) * 148,
    'ch3': np.random.randn(1000) * 152,
    'ch4': np.random.randn(1000) * 149,
    'f_ch1': np.random.randn(1000) * 50,
})

# Metadata
metadata = {
    'fs': 46875.0,
    'devFirmwareVersion': 2,
    'deviceSerial': '123456',
}
```

## Coverage Areas

### ✅ Tested
- Record creation and serialization
- Excel save/load with all sheets
- Incremental updates (partial vs. full)
- Helper functions with real data
- Error handling (failures, missing files)
- Integration with orchestration layer
- Multi-cycle updates
- Record replacement logic

### 🔄 Future Enhancements
- Performance tests (large datasets)
- Concurrent access tests (multiple processes)
- Malformed Excel file recovery
- Version migration tests (schema changes)
- More complex integration scenarios (full participant processing)

## Continuous Integration

### Adding to CI Pipeline

```yaml
# Example GitHub Actions workflow
- name: Run Processing Log Tests
  run: |
    pytest tests/test_processing_log.py \
           tests/test_processing_log_integration.py \
           -v --cov=src.orchestration.processing_log
```

## Debugging Failed Tests

### Common Issues

1. **Import Errors**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version compatibility

2. **File Permission Errors**
   - Tests use `tmp_path` which should have write permissions
   - On some systems, may need to check temp directory settings

3. **Excel File Issues**
   - Verify `openpyxl` is installed: `pip install openpyxl==3.1.3`
   - Check that no other process has Excel files open

### Running Individual Tests

```bash
# Run one test with verbose output
pytest tests/test_processing_log.py::TestManeuverProcessingLog::test_save_to_excel -v -s

# Add debugging with pdb
pytest tests/test_processing_log.py -k "test_save_to_excel" --pdb
```

## Test Maintenance

### Adding New Tests

When adding new functionality to the processing log system:

1. Add unit test to `test_processing_log.py`
2. Add integration test if it touches orchestration layer
3. Update standalone validation if it's a core feature
4. Update this documentation

### Example: Adding New Record Type

```python
# 1. Add to test_processing_log.py
class TestNewRecordType:
    def test_create_record(self):
        record = NewRecord(field="value")
        assert record.field == "value"

    def test_to_dict(self):
        record = NewRecord(field="value")
        data = record.to_dict()
        assert data["Field"] == "value"

# 2. Add to log tests
def test_add_new_record(self, tmp_path):
    log = ManeuverProcessingLog(...)
    log.add_new_record(NewRecord(...))
    assert len(log.new_records) == 1

# 3. Test save/load
def test_new_record_survives_roundtrip(self, tmp_path):
    # Create, save, load, verify
    ...
```

## Summary

The processing log system has **35+ automated tests** covering:
- ✅ All data models and their methods
- ✅ Excel save/load functionality
- ✅ Helper functions and data extraction
- ✅ Incremental update behavior
- ✅ Integration with orchestration layer
- ✅ Error handling and edge cases

**Run tests regularly** to ensure future changes don't break functionality:

```bash
# Quick check
python tests/test_processing_log_standalone.py

# Full test suite
pytest tests/test_processing_log*.py -v
```
