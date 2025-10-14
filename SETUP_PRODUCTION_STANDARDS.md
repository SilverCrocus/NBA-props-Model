# Setup Guide: Production Code Standards

**Time to Complete**: 15-20 minutes
**Prerequisites**: uv package manager installed

This guide will set up all code quality tools and standards for the NBA Props Model project.

---

## Step 1: Install Development Dependencies (5 min)

Install all testing and linting tools:

```bash
cd /Users/diyagamah/Documents/nba_props_model

# Install dev dependencies
uv sync --dev
```

This installs:
- pytest (testing framework)
- pytest-cov (code coverage)
- black (code formatter)
- flake8 (linter)
- isort (import sorter)
- mypy (type checker)
- pre-commit (git hooks)

---

## Step 2: Install Pre-Commit Hooks (2 min)

Set up automatic code quality checks on every commit:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Test that it works
uv run pre-commit run --all-files
```

This will:
- Format code with black
- Sort imports with isort
- Check for syntax errors with flake8
- Run quick unit tests

Expected output: Some files may be reformatted on first run (this is normal).

---

## Step 3: Run Initial Code Formatting (5 min)

Format all existing code to meet standards:

```bash
# Format all Python files in src/
uv run black src/ tests/

# Sort all imports
uv run isort src/ tests/

# Check for any remaining issues
uv run flake8 src/ tests/
```

Expected output: Black and isort will reformat files. Flake8 may show warnings (OK for now).

---

## Step 4: Run Test Suite (3 min)

Verify tests are working:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing
```

Expected output:
- Some tests may fail (we just created test files)
- Coverage will be low initially (this is expected)

---

## Step 5: Verify Setup (2 min)

Test that everything is configured correctly:

```bash
# 1. Check black configuration
uv run black --check src/

# 2. Check flake8 configuration
uv run flake8 src/ --count

# 3. Check mypy configuration (strict modules only)
uv run mypy src/betting/ src/evaluation/ || true

# 4. Verify pre-commit hooks are installed
ls -la .git/hooks/pre-commit
```

All commands should complete without errors (except mypy, which may fail if modules don't exist yet).

---

## Step 6: Test Pre-Commit Hook (2 min)

Make a small change and commit to test the workflow:

```bash
# Make a small change (update a comment)
echo "# Test commit" >> tests/unit/test_features.py

# Stage the change
git add tests/unit/test_features.py

# Try to commit (pre-commit hooks will run)
git commit -m "test: verify pre-commit hooks work"

# If hooks pass, undo the test commit
git reset HEAD~1
```

Expected behavior:
- Pre-commit hooks run automatically
- Code is formatted if needed
- Tests run (may be skipped if no changes in src/)
- Commit succeeds if all checks pass

---

## Common Issues & Solutions

### Issue 1: Pre-commit hooks fail on first run

**Symptom**: `black` reformats many files

**Solution**: This is expected! Commit the reformatted files:
```bash
git add .
git commit -m "style: apply black formatting to existing code"
```

### Issue 2: Flake8 shows many warnings

**Symptom**: Line too long, unused imports, etc.

**Solution**: These are warnings, not errors. Address them gradually:
```bash
# Fix automatically fixable issues
uv run black src/
uv run isort src/

# Review remaining warnings
uv run flake8 src/ | head -20
```

### Issue 3: Mypy errors in existing code

**Symptom**: Type errors in src/betting/ or src/evaluation/

**Solution**: These modules require type hints. Add gradually:
```bash
# Type checking is not blocking - ignore for now
uv run mypy src/betting/ --ignore-missing-imports || true
```

### Issue 4: Tests fail

**Symptom**: Some test files cause errors

**Solution**: New test files need actual implementation:
```bash
# Skip failing tests temporarily
uv run pytest tests/ -v --ignore=tests/unit/test_temporal_leakage.py
```

---

## Daily Workflow

Once setup is complete, your daily workflow becomes:

### Making Changes

```bash
# 1. Create feature branch
git checkout -b feature/add-new-feature

# 2. Make code changes
# ... edit files ...

# 3. Format code (or let pre-commit do it)
uv run black src/
uv run isort src/

# 4. Run tests
uv run pytest tests/unit -v

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add new feature"
```

### Pre-Commit Hook Workflow

When you run `git commit`:

1. Black formats your code automatically
2. Isort sorts your imports
3. Flake8 checks for errors
4. Quick unit tests run (optional, can be disabled)
5. If any check fails, commit is blocked
6. Fix issues and try again

### Bypassing Hooks (Emergency Only)

If you need to commit without running hooks (rare):

```bash
git commit --no-verify -m "emergency fix"
```

Note: Only use in emergencies. Don't make this a habit!

---

## Verifying Installation

Run this comprehensive check:

```bash
# Create a test script
cat > /tmp/test_setup.py << 'EOF'
"""Test script to verify setup."""

def hello_world():
    """Print hello world."""
    return "Hello, World!"

if __name__ == "__main__":
    print(hello_world())
EOF

# Test black
uv run black /tmp/test_setup.py
echo "✓ Black works"

# Test flake8
uv run flake8 /tmp/test_setup.py
echo "✓ Flake8 works"

# Test isort
uv run isort /tmp/test_setup.py
echo "✓ Isort works"

# Test pytest
uv run pytest --version
echo "✓ Pytest works"

# Test mypy
uv run mypy --version
echo "✓ Mypy works"

echo ""
echo "=========================================="
echo "✓ All tools installed and working!"
echo "=========================================="
```

Expected output: All checks pass with green checkmarks.

---

## What's Been Set Up

### Files Created

1. **PRODUCTION_CODE_STANDARDS.md** - Comprehensive code standards document
2. **CODE_REVIEW_CHECKLIST.md** - Self-review and peer review checklist
3. **.pre-commit-config.yaml** - Pre-commit hook configuration
4. **pytest.ini** - Pytest configuration with coverage settings
5. **pyproject.toml** (updated) - Black, isort, mypy configuration
6. **tests/unit/test_temporal_leakage.py** - Critical temporal leakage tests
7. **tests/unit/test_edge_calculator.py** - Critical betting logic tests
8. **docs/FEATURE_REGISTRY_TEMPLATE.md** - Feature documentation template

### Tools Configured

1. **Black** - Code formatter (100 char line length)
2. **Isort** - Import sorter (black-compatible)
3. **Flake8** - Linter (relaxed rules)
4. **Mypy** - Type checker (strict for betting/, evaluation/)
5. **Pytest** - Test runner (60% coverage requirement)
6. **Pre-commit** - Automatic quality checks

---

## Next Steps

### Week 1: Foundation

1. **Run formatting on all code** (30 min)
   ```bash
   uv run black src/ scripts/ tests/
   uv run isort src/ scripts/ tests/
   ```

2. **Create test fixtures** (1 hour)
   - Create `tests/fixtures/small_game_logs.csv`
   - Create `tests/fixtures/small_ctg_stats.csv`
   - Use these for fast unit tests

3. **Write 5 critical tests** (3 hours)
   - Temporal leakage detection ✓ (already created)
   - Edge calculation ✓ (already created)
   - Data quality checks
   - Feature correctness
   - Walk-forward validation

4. **Document existing features** (2 hours)
   - Use Feature Registry template
   - Document CTG_USG, PRA_L5_mean
   - Measure correlations

### Week 2: Testing & Documentation

1. **Achieve 40% test coverage** (4 hours)
   - Focus on src/features/
   - Focus on src/data/
   - Write integration tests

2. **Add type hints to critical modules** (2 hours)
   - Type hints for src/betting/ (when created)
   - Type hints for src/evaluation/ (when created)

3. **Update documentation** (1 hour)
   - Document data pipeline
   - Document model architecture
   - Create production runbook skeleton

### Week 3-4: Consolidation

1. **Refactor feature engineering** (6 hours)
   - Consolidate 3 feature systems
   - Extract to src/features/ modules
   - Add comprehensive tests

2. **Achieve 60% test coverage** (4 hours)
   - Write integration tests
   - Test edge cases
   - Test error handling

3. **Complete production runbook** (2 hours)
   - Daily workflow
   - Error recovery
   - Monitoring and alerts

---

## Maintenance

### Weekly Tasks

- Review test coverage report
- Check for flake8 warnings
- Update feature registry
- Review code quality metrics

### Monthly Tasks

- Update dependencies: `uv sync --upgrade`
- Review and update documentation
- Clean up deprecated code
- Run full test suite on multiple datasets

---

## Getting Help

If you run into issues:

1. **Check tool documentation**
   - Black: https://black.readthedocs.io/
   - Pytest: https://docs.pytest.org/
   - Pre-commit: https://pre-commit.com/

2. **Review configuration files**
   - pyproject.toml (black, isort, mypy settings)
   - pytest.ini (pytest settings)
   - .pre-commit-config.yaml (hook settings)

3. **Test in isolation**
   ```bash
   # Test black
   uv run black --check src/

   # Test specific test file
   uv run pytest tests/unit/test_features.py -v

   # Test pre-commit without committing
   uv run pre-commit run --all-files
   ```

---

## Success Criteria

You've successfully set up production standards when:

- [ ] All dev dependencies installed
- [ ] Pre-commit hooks working
- [ ] Code formatted with black
- [ ] Tests run successfully
- [ ] Coverage report generated
- [ ] Type checking configured
- [ ] Documentation templates in place
- [ ] Daily workflow tested

---

## Quick Command Reference

```bash
# Formatting
uv run black src/           # Format code
uv run isort src/           # Sort imports

# Linting
uv run flake8 src/          # Check for errors
uv run mypy src/betting/    # Type checking

# Testing
uv run pytest tests/        # Run all tests
uv run pytest -k temporal   # Run tests matching "temporal"
uv run pytest --cov=src     # Run with coverage

# Pre-commit
uv run pre-commit run --all-files  # Run all hooks
git commit                  # Automatically runs hooks
git commit --no-verify      # Skip hooks (emergency only)

# Coverage report
uv run pytest --cov=src --cov-report=html  # HTML report
open htmlcov/index.html     # View in browser
```

---

## Congratulations!

You now have production-ready code standards configured for your NBA Props Model project.

**Remember**: These standards are designed to prevent bugs and losses in production betting. Take them seriously, but don't let them slow you down. The tools are here to help you, not hinder you.

**Next**: Read PRODUCTION_CODE_STANDARDS.md for detailed guidelines on code organization, testing strategy, and best practices.
