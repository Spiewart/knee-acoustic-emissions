# AI Assistant Instructions for Acoustic Emissions Processing

## Quick Reference

**Project**: Acoustic emissions processing for biomechanical research
**Tech Stack**: Python 3.12+, Pydantic, Pandas, NumPy, SciPy, Matplotlib
**Virtual Env**: `.venv` (requires `requirements.txt` and `dev-requirements.txt`)

---

## Key Points for AI Assistants

### BiomechanicsMetadata Validation Rules ⚠️

This is critical for data integrity:

- **Walk maneuvers**: `speed` and `pass_number` are **required** (non-None)
- **Sit-to-stand & Flexion-extension**: `speed` and `pass_number` must be **None**

Validators enforce these rules in `models.py`. Always respect these constraints.

### Project Structure

```
acoustic_emissions_processing/
├── .venv/                              # Virtual environment
├── models.py                           # Pydantic data models (✓ Recently updated)
├── process_biomechanics.py             # Biomechanics processing
├── process_participant_directory.py    # Orchestration
├── parse_acoustic_file_legend.py       # Audio metadata parsing
├── sync_audio_with_biomechanics.py     # Data synchronization
├── read_audio_board_file.py            # .bin file parser
├── requirements.txt                    # Runtime dependencies
├── dev-requirements.txt                # Dev tools (pytest, black, mypy, etc.)
└── tests/                              # Test suite (45 tests, all passing)
```

### When Making Changes

1. **Update models**: Always update both the Pydantic model AND any functions that instantiate it
2. **Test coverage**: Run full test suite with `pytest tests/ -v` before finalizing
3. **Type hints**: Use type annotations for all function signatures
4. **Validators**: Add Pydantic validators for complex validation logic
5. **Docstrings**: Include Args, Returns, and Raises sections

### Current Implementation Status

✅ **Completed**:
- BiomechanicsMetadata with conditional validation
- Walk maneuver processing with full test coverage
- Audio-biomechanics synchronization

⏳ **TODO** (marked in code):
- Extend `import_biomechanics_recordings()` to handle sit_to_stand and flexion_extension
- Add start-time detection for non-walking maneuvers
- Improve DataFrame validation with more specific column checks

### Virtual Environment

**Runtime**: numpy, openpyxl, pandas, scipy, matplotlib, pydantic
**Dev**: black, isort, pytest, mypy, flake8

Activate with: `source .venv/bin/activate` (macOS/Linux) or `.\.venv\Scripts\Activate.ps1` (PowerShell)

### Testing

All 45 tests pass. Before committing:

```bash
pytest tests/ -v              # Full suite
pytest tests/test_smoke.py    # Quick smoke test
pytest tests/test_process_biomechanics.py  # Biomechanics-specific
```

**CRITICAL**: When validating code changes:
- Use `mcp_pylance_mcp_s_pylanceRunCodeSnippet` with `subprocess.run()` to execute tests
- Always `capture_output=True, text=True, timeout=60`
- Print full `result.stdout` to terminal so user sees test results
- Never use `run_in_terminal` (it hangs on environment activation)

### Code Style

- **Format**: `black` (line length: 88)
- **Imports**: `isort`
- **Type check**: `mypy` (prefer type hints)
- **Lint**: `flake8`

---

## AI Assistant Refactoring Standards

### When Refactoring or Modifying Code

1. **Type Hints Must Be Explicit**:
   - Use `Literal["value1", "value2"]` for constrained string values
   - Specify exact return types (e.g., `Literal["walk", "sit_to_stand", "flexion_extension"]`)
   - Never use generic `str` when values are constrained
   - Example: ❌ `def get_speed() -> str` → ✅ `def get_speed() -> Literal["slow", "normal", "fast"]`

2. **Extract Helper Functions**:
   - Complex logic should be extracted into focused helper functions
   - Prefix private functions with `_`: `_extract_maneuver_from_uid()`
   - Use clear, descriptive names
   - Keep functions single-responsibility

3. **Use Mapping Dictionaries**:
   - For value lookups, use explicit mapping dictionaries
   - Example: `maneuver_map = {"walk": "walk", "sittostand": "sit_to_stand"}`
   - Include `.get()` with error handling instead of direct key access

4. **Error Handling**:
   - Provide descriptive error messages
   - Example: ❌ `raise ValueError("Invalid")` → ✅ `raise ValueError(f"Unknown speed code '{code}' in pass info")`
   - Validate assumptions and fail fast

5. **Documentation**:
   - Include comprehensive docstrings
   - Format: Brief description, Args, Returns, Raises
   - Document edge cases and constraints
   - Example:
     ```python
     def get_biomechanics_metadata(uid: str) -> BiomechanicsMetadata:
         """Extract maneuver, pass number, and speed from UID.

         Args:
             uid: The unique identifier string

         Returns:
             BiomechanicsMetadata with maneuver, speed, and pass_number

         Raises:
             ValueError: If UID format is invalid or speed code not recognized
         """
     ```

### Validation Checklist After Code Changes

- [ ] All type hints are explicit (use `Literal` for constrained values)
- [ ] Complex logic extracted into private helper functions (`_function_name`)
- [ ] Mapping dictionaries used for lookups
- [ ] Error messages are descriptive
- [ ] Docstrings include Args, Returns, Raises
- [ ] **Full test suite passes** (45 tests)
- [ ] **Test output printed to terminal for user**

---

## Code Style

### Data Format Notes

- **UID Format**: `Study{ID}_{Maneuver}{Num}_{SpeedPass}_{Filt}` (e.g., Study123_Walk0001_NSP1_Filt)
- **Speed Codes**: SS=Slow, NS=Normal, FS=Fast
- **DataFrame TIME column**: Timedelta relative to start (not absolute)
- **Audio Channels**: ch1, ch2, ch3, ch4 (always 4 channels)
- **Microphones**: Always indexed 1, 2, 3, 4 (no 0-indexing)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | `source .venv/bin/activate && pip install -r requirements.txt` |
| Validation errors in BiomechanicsMetadata | Check maneuver type; walk needs speed+pass_number, others need None |
| DataFrame column missing | Verify source file structure matches expected columns |
| Tests fail | Run full suite to see if changes broke other tests |

---

## References

- [Pydantic Docs](https://docs.pydantic.dev/)
- [Pandas Docs](https://pandas.pydata.org/)
- [SciPy Signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
