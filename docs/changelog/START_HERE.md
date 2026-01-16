# ⚠️ IMPORTANT: Changelog Location Convention

**All change documentation must be stored in `/docs/changelog/` - NOT in the root directory.**

## Quick Rules for All Future Sessions

### ✅ DO THIS:
- Save change docs to: `docs/changelog/{type}/YYYYMMDD_name.md`
- Example: `docs/changelog/patch/20260115_feature_name.md`

### ❌ DON'T DO THIS:
- Save to root: `FEATURE_NAME.md` ← WRONG
- Save to root: `CHANGES.md` ← WRONG
- Save to root: `FIX_SUMMARY.md` ← WRONG

## The Four Change Types

| Folder | Use For | Examples |
|--------|---------|----------|
| `patch/` | Bug fixes, minor improvements | `20260115_filter_fix.md` |
| `feature/` | New features, major additions | `20260115_new_visualization.md` |
| `bugfix/` | Critical bugs (rarely used) | `20260115_data_loss_fix.md` |
| `refactor/` | Code restructuring | `20260115_module_reorganization.md` |

## File Naming
- Format: `YYYYMMDD_descriptive_name.md`
- Use underscores (not spaces)
- Keep concise but meaningful
- Dates sort automatically!

## Where to Find Instructions

1. **Quick Start**: This file (`docs/changelog/START_HERE.md`)
2. **Template & Details**: `docs/changelog/README.md`
3. **Comprehensive Guide**: `docs/changelog/GUIDE.md`
4. **Index of Changes**: `docs/changelog/INDEX.md`
5. **AI Instructions**: `ai_instructions.md` (search for "Change Documentation")

## One-Minute Process

```bash
# 1. Make your changes and run tests

# 2. Choose type: patch, feature, bugfix, or refactor

# 3. Create the file with today's date
docs/changelog/{TYPE}/YYYYMMDD_your_name.md

# 4. Fill in the template:
# - What was the problem?
# - How did you fix it?
# - What files changed?
# - How many tests pass?
# - Any breaking changes?

# 5. Done! Don't create files in root directory
```

## Example Entry

**File location**: `docs/changelog/patch/20260115_sync_fix.md`

```markdown
# Fixed Synchronization Validation with Filters

**Date**: 2026-01-15
**Type**: patch
**Affected Components**: src/orchestration/participant.py
**Test Coverage**: 68 tests (9 new + 59 existing)

## Problem
When using --knee or --maneuver filters, validation would fail
because it expected all maneuvers to have been processed.

## Solution
Added filter-aware validation that skips full checks when
filters are applied.

## Changes Made
- participant.py (lines 1631-1640): Added filter logic
- test_entrypoint_processing.py: Added filter validation test

## Test Results
✅ 68 tests pass

## Backward Compatible
Yes - default behavior unchanged

## Impact
- Fixes cascading with filters
- No breaking changes
```

## For AI Assistants Reading This Later

🤖 This is a standing instruction that applies to ALL future sessions:

**ALWAYS store change documentation in `/docs/changelog/{type}/`**

1. After making significant code changes, check `/docs/changelog/README.md` for the template
2. Create a file with naming format: `YYYYMMDD_descriptive_name.md`
3. Save it to the appropriate subfolder: `patch/`, `feature/`, `bugfix/`, or `refactor/`
4. Never save change documentation to the root directory
5. If unsure about type, see the table above or check `GUIDE.md`

This keeps the root directory clean and makes changes discoverable and organized!

## Questions?

See these files in order:
1. This file (you're reading it!)
2. `docs/changelog/README.md` - Templates and format
3. `docs/changelog/GUIDE.md` - Comprehensive guide
4. `ai_instructions.md` - AI-specific instructions

---

**Established**: 2026-01-15
**Status**: Active / In Use
**Last Reviewed**: 2026-01-15
