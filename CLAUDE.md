# CLAUDE.md — Project Guidelines for Claude Code

## Quick Reference

- **Virtual env**: `workon kae_processing`
- **Python**: 3.12+ required
- **Test command**: `python -m pytest tests/ -x -v --tb=short`
- **Lint**: `ruff check src/ cli/ tests/` and `mypy src/ cli/ --config-file=pyproject.toml`
- **Pre-commit**: `pre-commit run --all-files`

## Live Validation Requirement

**Any significant project-wide change must be validated against the sample directory.**
Unit tests alone are insufficient — live validation catches real file I/O, Excel parsing, and data edge cases.

1. **Process** a participant:

   ```bash
   python cli/process_directory.py \
     /Users/spiewart/kae_signal_processing_ml/sample_project_directory/ \
     --participant 1013 --entrypoint bin --persist-to-db
   ```

2. **Re-process** to verify idempotency:

   ```bash
   python cli/process_directory.py \
     /Users/spiewart/kae_signal_processing_ml/sample_project_directory/ \
     --participant 1013 --entrypoint bin --persist-to-db
   ```

Both runs must complete without errors.

## Key Rules

- All study-specific logic goes through `StudyConfig` protocol — never hardcode in pipeline modules
- Use test factory fixtures from `tests/conftest.py` — never create test data manually
- All config in `pyproject.toml` (ruff, mypy, pytest) + `.pre-commit-config.yaml`
- Time fields are `float` (seconds), not `timedelta`
- Pydantic models with `populate_by_name=True` accept both field name and alias
