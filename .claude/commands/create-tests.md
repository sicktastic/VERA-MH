Analyze test coverage and systematically improve it:

1. **Run Coverage Analysis**
   - Execute: `uv run pytest --cov=vera_mh --cov-report=term-missing --cov-report=json`
   - Parse coverage report
   - Identify modules with coverage below quality target (75%+)
   - Note: Project has dual coverage targets:
     - 30% = CI minimum (must pass for all code)
     - 75%+ = Quality target (goal for new/changed code)

2. **Prioritize Modules**
   Rank by importance:
   - **Critical**: Mental health safety code, prompt handling, data validation
   - **High**: LLM clients, conversation generation, evaluation
   - **Medium**: Utilities, configuration
   - **Low**: Scripts, examples

3. **Analyze Coverage Gaps**
   For each low-coverage module:
   - Identify uncovered lines
   - Determine why not covered (untested functions, edge cases, error paths)
   - Assess complexity and risk

4. **Present Coverage Report**
   Show user:
   ```
   Coverage Report:

   Critical (needs attention):
   - llm_clients/claude_llm.py: 45% (target: 75%+)
     Uncovered: error handling, retry logic

   High (below target):
   - generate_conversations/runner.py: 62%
     Uncovered: edge cases in async execution

   Medium (good):
   - utils/logging_utils.py: 82%
   ```

5. **Get User Approval**
   Ask which modules to create tests for:
   - All critical modules (recommended)
   - Specific modules
   - Custom selection

6. **Create Tests**
   For each approved module:
   - Use `/create-test` patterns
   - Analyze the module's functions
   - Create comprehensive test file
   - Include unit tests for all uncovered functions
   - Add edge case and error path tests
   - Use appropriate fixtures (MockLLM, tmp_path, etc.)

7. **Verify Improvement**
   - Re-run coverage: `uv run pytest --cov`
   - Show before/after comparison
   - Report: "Coverage improved from 65% → 78%"

## Example Workflow

User runs `/improve-coverage`:

```
Analyzing coverage...

Coverage Report (Current: 65%, Quality Target: 75%):

🔴 Critical - Needs Attention:
  llm_clients/claude_llm.py: 45% (↓ 30% from quality target)
    Uncovered: Lines 78-95 (error handling), 120-135 (retry logic)

  generate_conversations/conversation_simulator.py: 58% (↓ 17%)
    Uncovered: Lines 45-52 (early termination), 89-103 (async errors)

🟡 High Priority - Below Target:
  generate_conversations/runner.py: 68% (↓ 7%)
    Uncovered: Lines 156-170 (concurrent execution errors)

🟢 Good Coverage:
  utils/prompt_loader.py: 88%
  utils/model_config_loader.py: 92%

Create tests for critical modules? (yes/all/select/no)
> all

Creating tests for llm_clients/claude_llm.py...
✅ Created tests/unit/llm_clients/test_claude_llm.py (15 test cases)

Creating tests for conversation_simulator.py...
✅ Created tests/unit/test_conversation_simulator.py (12 test cases)

Creating tests for runner.py...
✅ Created tests/unit/test_runner.py (8 test cases)

Running tests to verify...
✅ All new tests pass

Re-analyzing coverage...
Coverage improved: 65% → 78% (+13%)

🎉 Target achieved! Critical modules now above 75%.
```
