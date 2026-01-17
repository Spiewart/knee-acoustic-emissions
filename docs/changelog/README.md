# AI Assistant Instructions: Changelog Management

## Overview
All change documentation should be organized in `/docs/changelog/` with type-specific subfolders rather than cluttering the root directory.

## Folder Structure

```
docs/changelog/
├── patch/           # Bug fixes, minor improvements to existing functionality
├── feature/         # New features, significant additions
├── bugfix/          # Critical bug fixes (use patch/ for most bug fixes)
├── refactor/        # Code refactoring, restructuring without behavioral changes
└── README.md        # Index of all changes
```

## When to Create a Change Document

Create a changelog document whenever you make changes that:
- Fix significant bugs
- Add new features
- Include substantial refactoring
- Change API or user-facing behavior
- Include notable test additions
- Span multiple files or components

**Do NOT create** for:
- Trivial fixes (single line changes)
- Inline documentation updates
- Minor formatting changes

## File Naming Convention

Use this format: `YYYYMMDD_descriptive_name.md`

Examples:
- `20260115_entrypoint_filter_processing.md`
- `20260115_stomp_detection_improvements.md`
- `20260114_audio_qc_refactor.md`

## Document Template

Use this template for all change documents:

```markdown
# Title of Change

**Date**: YYYY-MM-DD
**Type**: patch | feature | bugfix | refactor
**Affected Components**: file1.py, file2.py
**Test Coverage**: N tests (M new + K existing)

## Problem/Motivation
Brief description of what problem this solves or what feature it adds.

## Solution
How the problem is solved.

## Changes Made

### Files Modified
- **file.py** (lines X-Y): Description
- **tests/test_file.py** (NEW): Description

### Key Code Changes
Show important code snippets if applicable.

## Test Results

✅ **N tests pass** (M new + K existing)
- Brief summary of what tests verify

## Behavioral Changes

### Now Works/Works Better
Examples of new or improved behavior.

### Backward Compatible
Whether breaking changes exist.

## Impact Summary
- **Fixes/Adds**: What is fixed or added
- **Breaking changes**: None (or list them)
- **Reliability**: Impact on system reliability
```

## Usage Instructions

### For AI Assistants (this system prompt):
1. After making significant changes to the codebase, create a changelog document
2. Save it to `/docs/changelog/{type}/{YYYYMMDD}_{name}.md`
3. **Never** save change documentation to the root directory
4. Always include test results and backward compatibility notes
5. Use precise line numbers and file paths
6. Make documents self-contained (someone unfamiliar with the changes should understand them)

### For Users Reviewing Changes:
1. Check `/docs/changelog/` for recent changes
2. Filter by type (patch, feature, bugfix, refactor) based on your interest
3. Review chronologically via file dates
4. Look for breaking changes and test coverage

## Change Type Guidelines

### patch/
- Bug fixes with minimal scope
- Performance improvements
- Minor API adjustments
- Logic corrections

Examples:
- Fixed filter validation in participant processing
- Improved stomp detection accuracy
- Corrected off-by-one error in loop

### feature/
- New functionality added
- Significant API additions
- New commands or options
- Major capability enhancements

Examples:
- Added multi-method stomp detection
- Implemented cascading entrypoint logic
- New visualization capabilities

### bugfix/
- Critical bugs that break functionality
- Security issues
- Data corruption risks
- Use sparingly (prefer patch/ for most fixes)

Examples:
- Fixed data loss on sync failure
- Corrected audio channel misalignment
- Fixed memory leak in signal processing

### refactor/
- Code restructuring
- Architecture improvements
- No user-facing behavior changes
- Build/test infrastructure changes

Examples:
- Reorganized module structure
- Refactored validation logic
- Improved test organization

## README Maintenance

Maintain `/docs/changelog/README.md` with:
- Quick index of recent changes
- Links to all changelog documents
- Summary of major improvements by date range

## Examples

✅ **Good example document location:**
`/docs/changelog/patch/20260115_entrypoint_filter_processing.md`

❌ **Bad example locations (root directory):**
- `/ENTRYPOINT_FIX.md` ← Use docs/changelog instead
- `/FILTER_VALIDATION_CHANGES.md` ← Use docs/changelog instead

## Future Session Continuity

Future sessions should:
1. Check this file to understand the changelog organization
2. Always store change documentation in `/docs/changelog/{type}/`
3. Use the file naming convention
4. Follow the document template
5. Reference this document when creating new changelogs

This ensures all change documentation is:
- Organized and findable
- Consistent in format
- Time-tracked for version history
- Categorized by change type
- Kept out of the root directory

## 2026-01-17
- `feature/20260117_energy_ratio_validation.md` – Enforce recorded knee louder than contralateral
- `feature/20260117_energy_ratio_maintenance.md` – Maintenance guide for energy ratio validation
- `feature/20260117_consensus_clustering.md` – Replace median with clustering-based consensus method
