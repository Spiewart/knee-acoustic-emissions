# Changelog Index

## Recent Changes

### 2026-01-15

#### Patches
- **[20260115_entrypoint_filter_processing.md](patch/20260115_entrypoint_filter_processing.md)**
  - Fixed entrypoint cascading logic (bin → sync → cycles)
  - Fixed filter validation to allow partial processing
  - 68 tests pass (9 new + 59 existing)

## How to Use This Directory

**For finding recent changes**: Look at the date-based filenames (YYYYMMDD format) - newer dates appear alphabetically later.

**By change type**:
- **[patch/](patch/)** - Bug fixes and minor improvements
- **[feature/](feature/)** - New features and major additions
- **[bugfix/](bugfix/)** - Critical bugs (rarely used)
- **[refactor/](refactor/)** - Code restructuring

**For detailed guidance**:
- **[README.md](README.md)** - Templates and usage instructions
- **[GUIDE.md](GUIDE.md)** - Comprehensive guide for all scenarios

## Quick Reference

### To Find Changes
```bash
# List all recent patches
ls -1tr patch/

# Search for specific keywords
grep -r "stomp detection" .
grep -r "breaking change" .

# Count changes by type
find patch -name "*.md" | wc -l
find feature -name "*.md" | wc -l
```

### To Add a Change
1. Choose type: `patch`, `feature`, `bugfix`, or `refactor`
2. Create: `{type}/YYYYMMDD_descriptive_name.md`
3. Use template from [README.md](README.md)
4. Include problem, solution, files modified, tests, backward compatibility

### Document Template Quick Reference
```markdown
# Title of Change

**Date**: YYYY-MM-DD
**Type**: patch | feature | bugfix | refactor
**Affected Components**: file.py, ...
**Test Coverage**: N tests (M new + K existing)

## Problem/Motivation
...

## Solution
...

## Changes Made
...

## Test Results
✅ N tests pass

## Behavioral Changes
...

## Impact Summary
...
```

## Statistics

| Category | Count |
|----------|-------|
| Total Changes | 1 |
| Patches | 1 |
| Features | 0 |
| Bug Fixes | 0 |
| Refactors | 0 |

**Note**: Update this index as new changes are added. For comprehensive guidelines, see [GUIDE.md](GUIDE.md) or [README.md](README.md).

---

**Last Updated**: 2026-01-15
**Maintainer Note**: This is the official changelog location. Do NOT add documentation to the root directory. See [ai_instructions.md](../../ai_instructions.md) for AI assistant instructions.
