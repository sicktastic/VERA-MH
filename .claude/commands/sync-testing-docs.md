Audit and synchronize testing documentation across slash commands, agents, and codebase:

## Purpose

Ensure that testing configuration stays in sync as:
- New slash commands are added
- Existing commands are updated
- Repo testing patterns change
- Testing infrastructure evolves

## Workflow

### 1. Discover Testing Commands
- List all files in `.claude/commands/`
- Identify testing-related commands (test, create-test, improve-coverage, setup-dev, etc.)
- Note any new commands since last sync

### 2. Audit test-engineer Agent
- Read `.claude/agents/test-engineer.md`
- Check "Reference Documentation" section
- Compare listed commands against discovered commands
- Identify:
  - Missing commands (new ones not yet referenced)
  - Outdated descriptions (commands changed purpose)
  - Removed commands (still referenced but deleted)

### 3. Review Current Testing Patterns
- Read key test files to understand current patterns:
  - `tests/conftest.py` - Fixtures and shared setup
  - `tests/unit/test_*.py` - Unit test examples
  - `pyproject.toml` - Pytest configuration
- Identify if testing conventions have evolved
- Check if commands reflect current patterns

### 4. Check Command Consistency
For each testing command, verify:
- Does it reflect current repo structure?
- Are pytest commands up-to-date?
- Does coverage target match project standard?
- Are fixture names and markers consistent?

### 5. Present Findings
Show user:
```
Testing Documentation Audit:

✅ In Sync:
  - /test command reflects current pytest setup
  - /create-test patterns match tests/conftest.py fixtures

⚠️ Needs Update:
  - test-engineer agent missing reference to new /test-watch command
  - /improve-coverage still references 75% target (changed to 80% in pyproject.toml)

❌ Issues:
  - /setup-dev references old pytest-cov version
  - Agent "Reference Documentation" still lists deleted /test-coverage command
```

### 6. Get User Approval
Ask which updates to apply:
- All recommended updates
- Specific updates only
- Skip (just show report)

### 7. Apply Updates
For approved changes:
- Update agent "Reference Documentation" section
- Update command files to reflect current repo patterns
- Update CLAUDE.md if testing guidelines changed
- Update READMEs if command list changed

### 8. Verify Changes
- Show diff of what was changed
- Confirm all testing docs are now in sync
- Suggest running `/test` to verify changes don't break anything

## When to Run This Command

**Recommended frequency:**
- After adding new testing commands
- After major testing infrastructure changes
- Before creating PRs that involve testing changes
- Quarterly maintenance check

**Triggers:**
- "Added a new /test-xyz command, need to update agent"
- "Changed pytest configuration, update docs"
- "Not sure if testing docs are in sync"

## Example Output

```
Running testing documentation audit...

Discovered Commands:
  - /test
  - /create-test
  - /improve-coverage
  - /setup-dev
  - /test-watch (NEW)

Checking test-engineer agent...
  ⚠️ Missing reference to /test-watch command

Checking repo testing patterns...
  ✅ Commands reflect current pytest setup
  ✅ Coverage target consistent (75%)
  ✅ Fixtures match conftest.py

Recommendations:
1. Add /test-watch to agent "Reference Documentation"
   - Description: "Watch mode for continuous test running"

Apply updates? (yes/no/select)
> yes

Updating .claude/agents/test-engineer.md...
✅ Added /test-watch reference

All testing documentation is now in sync!
```
