# Testing Guidelines: Consolidated Fixture Pattern

## ⚠️ CRITICAL RULE

**DO NOT CREATE TEST DATA MANUALLY IN INDIVIDUAL TEST FILES!**

This project uses **consolidated fixture factories** in `tests/conftest.py` as the single source of truth for all test data creation. This pattern is MANDATORY for maintainability.

## Quick Reference

### ✅ CORRECT Pattern

```python
def test_synchronization(synchronization_factory):
    """Always use factory fixtures."""
    
    # Create test data with factory
    sync = synchronization_factory(
        audio_sync_time=5.0,
        knee="left",
        processing_status="success"
    )
    
    # Test your logic
    assert sync.audio_sync_time == 5.0
```

### ❌ INCORRECT Pattern

```python
def test_bad_pattern():
    """DO NOT DO THIS!"""
    
    # ❌ Creating test data manually
    from src.metadata import Synchronization
    sync = Synchronization(
        study="AOA",
        study_id=1001,
        # ... 50 lines of boilerplate
    )
```

## Why This Pattern is Mandatory

### Problem: Scattered Test Data Creation

Before consolidation (Phase 4), we had:
- 42 separate helper functions across 15+ test files
- Duplicate default values (inconsistent test behavior)
- No single source of truth for test data

**Result**: When we changed time fields from `timedelta` to `float` (Phase 5), we had to update 42 different locations. Took ~8 hours. High risk of introducing bugs.

### Solution: Consolidated Fixture Factories

After consolidation (Phase 5), we have:
- 4 factory functions in ONE file (`tests/conftest.py`)
- Consistent defaults across all tests
- Single source of truth

**Result**: Future metadata changes now update ONE place instead of 42. Estimated time savings: 80% reduction in test maintenance.

## Available Factory Fixtures

All factories are defined in [`tests/conftest.py`](../tests/conftest.py) and automatically available to all tests:

### synchronization_factory

Creates `Synchronization` instances with sensible defaults.

**Usage**:
```python
def test_sync_workflow(synchronization_factory):
    # Minimal override
    sync = synchronization_factory(sync_file_name="test.pkl")
    
    # Multiple overrides
    sync = synchronization_factory(
        audio_sync_time=5.0,
        bio_left_sync_time=10.0,
        sync_duration=120.0,
        knee="left",
        maneuver="walk",
        processing_status="success"
    )
```

**Default Values** (See `conftest.py` for complete list):
- `study="AOA"`, `study_id=1001`
- `audio_sync_time=5.0`, `bio_left_sync_time=10.0`
- `sync_duration=120.0` (float seconds)
- `knee="left"`, `maneuver="walk"`
- `processing_status="success"`

### synchronization_metadata_factory

Creates `SynchronizationMetadata` instances.

**Usage**:
```python
def test_metadata_validation(synchronization_metadata_factory):
    meta = synchronization_metadata_factory(
        audio_sync_time=10.0,
        sync_method="consensus"
    )
```

### audio_processing_factory

Creates `AudioProcessing` instances.

**Usage**:
```python
def test_audio_qc(audio_processing_factory):
    audio = audio_processing_factory(
        audio_file_name="recording.bin",
        sample_rate=46875.0,
        duration_seconds=120.0
    )
```

### movement_cycle_factory

Creates `MovementCycle` instances.

**Usage**:
```python
def test_cycle_extraction(movement_cycle_factory):
    cycle = movement_cycle_factory(
        cycle_number=1,
        cycle_duration_s=1.2,
        cycle_start_time=0.0,
        cycle_end_time=1.2
    )
```

## Time Field Format (Post Phase-5)

**CRITICAL**: All time fields use **float (seconds)**, NOT `timedelta`.

### ✅ CORRECT

```python
sync = synchronization_factory(
    audio_sync_time=5.0,        # Float seconds
    sync_duration=120.0,         # Float seconds
    consensus_time=5.2,          # Float seconds
    rms_time=5.1                 # Float seconds
)
```

### ❌ INCORRECT

```python
from datetime import timedelta

# WRONG! Will raise ValidationError
sync = synchronization_factory(
    audio_sync_time=timedelta(seconds=5.0),  # NO!
    sync_duration=timedelta(seconds=120.0)   # NO!
)
```

## Adding New Metadata Fields

When you add fields to metadata classes, follow this workflow:

### 1. Update Metadata Class

```python
# src/metadata.py
@dataclass
class Synchronization(SynchronizationMetadata):
    """Add your new field."""
    
    new_qc_parameter: Optional[float] = None
    
    @field_validator("new_qc_parameter")
    @classmethod
    def validate_new_parameter(cls, value):
        if value is not None and value < 0:
            raise ValueError("must be non-negative")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["New QC Parameter"] = self.new_qc_parameter
        return result
```

### 2. Update Factory in conftest.py

```python
# tests/conftest.py
@pytest.fixture
def synchronization_factory():
    def _create(**overrides):
        defaults = {
            # ... existing defaults
            "new_qc_parameter": 0.0,  # ← Add sensible default
        }
        defaults.update(overrides)
        return Synchronization(**defaults)
    return _create
```

### 3. Update Production Code

```python
# src/orchestration/processing_log.py
def create_sync_record_from_data(...):
    return Synchronization(
        new_qc_parameter=data.get("new_qc_parameter", 0.0),
        # ... other fields
    )
```

### 4. Tests Automatically Get New Field

```python
def test_with_new_field(synchronization_factory):
    # Works immediately with factory default
    sync = synchronization_factory()
    assert sync.new_qc_parameter == 0.0
    
    # Override when needed
    sync = synchronization_factory(new_qc_parameter=5.0)
    assert sync.new_qc_parameter == 5.0
```

**CRITICAL**: Do NOT create a new helper function in your test file. Update the factory!

## Common Patterns

### Testing Validation Rules

```python
def test_validation_error(synchronization_factory):
    """Test that invalid data raises errors."""
    
    with pytest.raises(ValidationError):
        synchronization_factory(
            audio_sync_time=-1.0  # Invalid: negative time
        )
```

### Testing with Minimal Data

```python
def test_minimal_sync(synchronization_factory):
    """Factory provides all required fields."""
    
    # Only override what you're testing
    sync = synchronization_factory(processing_status="error")
    
    # All other fields have working defaults
    assert sync.processing_status == "error"
    assert sync.study == "AOA"  # Default
```

### Testing Multiple Instances

```python
def test_multiple_records(synchronization_factory):
    """Create multiple test instances easily."""
    
    syncs = [
        synchronization_factory(pass_number=i, speed=speed)
        for i, speed in enumerate(["slow", "medium", "fast"], 1)
    ]
    
    assert len(syncs) == 3
    assert syncs[0].speed == "slow"
```

## Historical Context: Phase 5 Refactoring

### The Problem We Solved

Before Phase 5, time fields used `timedelta` objects. Each test file had its own helper function creating test data with `timedelta(seconds=X)`.

**Files with duplicate helpers**: 15+  
**Total helper functions**: 42+  
**Lines of duplicated code**: ~2000

### The Change

Phase 5 converted all time fields from `timedelta` to `float` (seconds).

**Without consolidation**: Would need to update 42 helper functions across 15+ files.  
**With consolidation**: Only updated 4 factories in `conftest.py`.

**Time saved**: ~8 hours  
**Bugs prevented**: Many (inconsistent conversions, missed `.total_seconds()` calls)

### The Lesson

**Technical debt compounds exponentially.** Every new test file that creates its own helper function makes future changes harder. The consolidated fixture pattern prevents this.

## Enforcement

This pattern is enforced through:

1. **Documentation**: This guide, README.md, ai_instructions.md, and conftest.py docstring
2. **Code Review**: Pull requests adding manual test data creation should be rejected
3. **AI Assistant Instructions**: Claude/GitHub Copilot trained to use factories
4. **Developer Onboarding**: New team members must read this guide

## Questions?

If you need a new factory or are unsure how to use existing ones:

1. Check the factory docstring in `tests/conftest.py`
2. Look at existing tests using the factory (grep for `factory` in tests/)
3. Ask the team - but the answer is almost always "use the factory"

## Summary

✅ **DO**: Use `synchronization_factory()` and related fixtures  
✅ **DO**: Override only what you need to test  
✅ **DO**: Update factories when adding metadata fields  
✅ **DO**: Use float (seconds) for all time fields  

❌ **DON'T**: Create `Synchronization()` directly in tests  
❌ **DON'T**: Make new helper functions in test files  
❌ **DON'T**: Use `timedelta` for time fields  
❌ **DON'T**: Duplicate default values across tests  

**Remember**: Consolidated fixtures = Maintainable tests = Happy developers! 🎉
