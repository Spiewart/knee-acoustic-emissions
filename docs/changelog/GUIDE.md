# Changelog Organization Guide

## Quick Start

### For AI Assistants
When you finish making significant changes:

1. **Create a document** in the appropriate subfolder:
   ```bash
   docs/changelog/{type}/{YYYYMMDD}_{name}.md
   ```

2. **Choose the right type**:
   - `patch/` - Bug fixes, minor improvements
   - `feature/` - New features, major additions
   - `bugfix/` - Critical bugs (use `patch/` for most fixes)
   - `refactor/` - Code restructuring, no behavior changes

3. **Use the template** from `docs/changelog/README.md`

4. **Never save to root** - Always use the changelog subfolder

### For Users
- Browse `docs/changelog/{type}/` to find changes of interest
- Sort by date: most recent files are at the end (YYYYMMDD format)
- Read `docs/changelog/README.md` for full documentation
- Check `ai_instructions.md` section on "Change Documentation & Changelog"

## Example Workflow

### Scenario: You fixed a bug in stomp detection
```
1. Make code changes to src/synchronization/sync.py
2. Write and run tests
3. Create: docs/changelog/patch/20260115_stomp_detection_fix.md
4. In that file, document the problem, solution, and tests
5. Done! No need to create files in root directory
```

### Scenario: You added a new feature
```
1. Implement the feature across multiple files
2. Add comprehensive tests
3. Create: docs/changelog/feature/20260115_new_visualization.md
4. Link to the files you modified
5. Document backward compatibility
```

## File Organization

### What Goes in Each Folder

**patch/** - Bugs and minor improvements
```
20260115_stomp_detection_accuracy.md
20260114_validation_filter_fix.md
20260113_memory_optimization.md
```

**feature/** - New capabilities and major additions
```
20260115_multi_method_stomp_detection.md
20260110_realtime_processing.md
20260105_new_qc_metrics.md
```

**bugfix/** - Critical issues (rarely used)
```
20260115_data_loss_fix.md
20260110_security_vulnerability.md
```

**refactor/** - Code restructuring
```
20260115_module_reorganization.md
20260112_test_infrastructure.md
20260108_validation_logic_refactor.md
```

## Document Filename Convention

**Format**: `YYYYMMDD_descriptive_name.md`

**Rules**:
- Use date in YYYYMMDD format (ISO 8601)
- Use lowercase with underscores
- Keep name descriptive but concise (20-40 characters)
- Make it searchable and meaningful

**Good Examples**:
- `20260115_entrypoint_filter_processing.md`
- `20260115_stomp_detection_improvements.md`
- `20260114_audio_qc_refactor.md`

**Bad Examples**:
- `fix.md` (not descriptive)
- `changes_made_today.md` (vague)
- `January15_update.md` (not machine-sortable)

## What Should Include a Changelog Entry?

### ✅ Create an Entry For:
- Bug fixes (especially significant ones)
- New features
- Breaking API changes
- Major refactoring
- Notable performance improvements
- Test infrastructure changes
- Documentation restructuring

### ❌ Don't Create an Entry For:
- Single-line fixes
- Comment updates
- Formatting changes
- Trivial typo fixes
- Minor docstring updates
- Small variable renamings

## How to Find Changes

### By Date (Most Recent First)
```bash
ls -1rt docs/changelog/patch/   # Oldest first
ls -1tr docs/changelog/patch/   # Newest first
```

### By Type
```bash
find docs/changelog/patch/ -name "*.md"      # All patches
find docs/changelog/feature/ -name "*.md"    # All features
```

### By Keyword
```bash
grep -r "stomp detection" docs/changelog/
grep -r "breaking change" docs/changelog/
```

## Maintenance

### Archiving Old Changes
As the changelog grows, periodically archive older changes:
```bash
# Create yearly archives
docs/changelog/archive/2025/
docs/changelog/archive/2026/
```

### Index Updates
The README.md in docs/changelog/ should maintain:
- List of recent significant changes
- Links to major features/fixes
- Quick statistics (bugs fixed, features added)

### Version Milestones
For each release, create a summary:
```bash
docs/changelog/releases/v1.0.0_summary.md
```

## Integration with Version Control

### Recommended Git Workflow
```bash
# Feature branch
git checkout -b feature/new-capability

# Make changes
# Add tests
# Create changelog entry: docs/changelog/feature/20260115_new_capability.md

# Commit
git add .
git commit -m "feat: add new capability

- Added new feature XYZ
- See docs/changelog/feature/20260115_new_capability.md"

# PR includes both code and changelog
```

### Commit Message Format
Include a reference to the changelog entry:
```
feat: add feature XYZ
See: docs/changelog/feature/20260115_new_feature.md

- Implementation details
- Test results: 50 tests pass
```

## Best Practices

### 1. Write for the Future
Assume the reader doesn't know the context. Be thorough but concise.

### 2. Include Line Numbers
Reference specific code changes with line numbers to help readers locate changes.

### 3. Document Backwards Compatibility
Always note if a change breaks existing functionality.

### 4. Link to Tests
Mention test coverage and link to test files when relevant.

### 5. Use Clear Sections
Follow the standard template:
- Problem/Motivation
- Solution
- Files Modified
- Test Results
- Behavioral Changes
- Impact Summary

### 6. Be Precise
- Use exact file paths
- Include line numbers or ranges
- Reference specific functions/classes
- Quote important code snippets

## Examples of Well-Written Entries

See these actual entries:
- `docs/changelog/patch/20260115_entrypoint_filter_processing.md` - Good example of patch documentation with problem, solution, tests, and backward compatibility

## Common Mistakes to Avoid

❌ Saving to root directory instead of `/docs/changelog/`
❌ Using non-YYYYMMDD date format
❌ Vague or unclear problem descriptions
❌ No test coverage information
❌ Forgetting backward compatibility notes
❌ Creating entry for trivial changes
❌ Using spaces in filename instead of underscores
❌ Forgetting to update file paths in future documentation

## Questions?

Refer to:
1. This guide (`docs/changelog/GUIDE.md`)
2. Template in `docs/changelog/README.md`
3. Example entry: `docs/changelog/patch/20260115_entrypoint_filter_processing.md`
4. AI instructions: `ai_instructions.md` (search for "Change Documentation")
