# Production Standards - Quick Reference Card

**One-page reference for daily development**

Print this out and keep it at your desk!

---

## Setup (One Time Only)

```bash
cd /Users/diyagamah/Documents/nba_props_model
uv sync --dev
uv run pre-commit install
```

---

## Daily Commands

### Code Formatting
```bash
uv run black src/              # Format code
uv run isort src/              # Sort imports
```

### Testing
```bash
uv run pytest tests/unit -v   # Quick unit tests
uv run pytest tests/ --cov    # Full tests with coverage
```

### Linting
```bash
uv run flake8 src/            # Check for errors
uv run mypy src/betting/      # Type check critical modules
```

### Git Workflow
```bash
git checkout -b feature/name   # Create branch
# ... make changes ...
git add .
git commit -m "feat: message"  # Pre-commit hooks run automatically
git push origin feature/name
```

---

## Before Merging to Main

- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] Coverage > 60%: Check coverage report
- [ ] Code formatted: `uv run black --check src/`
- [ ] No critical errors: `uv run flake8 src/`
- [ ] Code reviewed: Use CODE_REVIEW_CHECKLIST.md

---

## Critical Code Patterns

### Creating Temporal Features (PREVENT LEAKAGE!)
```python
# ALWAYS use .shift(1) to exclude current game
df['feature'] = (
    df.groupby('PLAYER_ID')['stat']
    .shift(1)  # CRITICAL!
    .rolling(window=5)
    .mean()
)
```

### Handling Missing Data
```python
# ALWAYS be explicit about defaults
try:
    value = load_data(player, season)
except FileNotFoundError:
    logger.warning(f"Data not found, using default")
    value = DEFAULT_VALUE
```

### Train/Test Split
```python
# ALWAYS use chronological splits
train = df[df['GAME_DATE'] <= train_end]
test = df[df['GAME_DATE'] > train_end]

# VERIFY no overlap
assert train['GAME_DATE'].max() < test['GAME_DATE'].min()
```

---

## Test Markers

```bash
pytest -m critical          # Run critical tests only
pytest -m temporal          # Run temporal leakage tests
pytest -m "not slow"        # Skip slow tests
```

---

## When to Use What

| Task | Tool | Command |
|------|------|---------|
| Format code | black | `uv run black src/` |
| Sort imports | isort | `uv run isort src/` |
| Check errors | flake8 | `uv run flake8 src/` |
| Type check | mypy | `uv run mypy src/betting/` |
| Run tests | pytest | `uv run pytest tests/` |
| Check coverage | pytest | `uv run pytest --cov=src` |

---

## File Quick Reference

| Need | File |
|------|------|
| Detailed standards | PRODUCTION_CODE_STANDARDS.md |
| Review checklist | CODE_REVIEW_CHECKLIST.md |
| Setup instructions | SETUP_PRODUCTION_STANDARDS.md |
| Summary overview | PRODUCTION_STANDARDS_SUMMARY.md |
| This card | QUICK_REFERENCE.md |

---

## Coverage Targets

- src/features/: **85%+**
- src/evaluation/: **85%+**
- src/betting/: **85%+**
- src/data/: **70%+**
- Overall: **65%+**

---

## Critical Tests (Must Pass)

1. test_temporal_leakage.py - Prevents using future data
2. test_edge_calculator.py - Ensures betting logic correct
3. test_data_quality.py - Validates data integrity
4. test_features.py - Verifies feature calculations
5. test_walk_forward.py - Validates time series splits

---

## Common Mistakes to Avoid

1. ❌ Rolling average without .shift(1)
2. ❌ Using current season stats to predict current season
3. ❌ Random train/test splits on time series data
4. ❌ Merging without deduplication
5. ❌ Hardcoding paths or credentials
6. ❌ Using print() instead of logging
7. ❌ Committing without running tests

---

## Commit Message Format

```
feat: add new feature
fix: fix bug description
test: add tests for X
docs: update documentation
refactor: reorganize code
perf: improve performance
```

---

## Emergency Procedures

### Tests Failing
```bash
# Run specific test to debug
uv run pytest tests/unit/test_file.py::test_name -v -s

# Check what changed
git diff

# Revert if needed
git checkout -- file.py
```

### Pre-commit Hook Blocking
```bash
# See what's failing
uv run pre-commit run --all-files

# Emergency bypass (use sparingly!)
git commit --no-verify -m "message"
```

### Coverage Too Low
```bash
# See what's missing
uv run pytest --cov=src --cov-report=term-missing

# Focus on untested files
uv run pytest tests/ --cov=src/features/ --cov-report=html
open htmlcov/index.html
```

---

## Questions?

| Question | Answer |
|----------|--------|
| How do I add a new feature? | 1. Document in registry 2. Write test 3. Implement 4. Verify no leakage |
| How do I know if I have leakage? | Run test_temporal_leakage.py |
| What coverage do I need? | 60%+ overall, 85%+ on critical modules |
| When should I refactor? | Incrementally, when adding features |
| Can I skip tests? | No for critical tests. Rare exceptions for non-critical. |

---

## Help Commands

```bash
# See all pytest options
uv run pytest --help

# See black configuration
uv run black --help

# See coverage report
uv run pytest --cov=src --cov-report=term-missing

# List all tests
uv run pytest --collect-only

# Run tests matching pattern
uv run pytest -k "temporal" -v
```

---

## Weekly Checklist

Every Friday:
- [ ] Review test coverage
- [ ] Check for flake8 warnings
- [ ] Update feature registry
- [ ] Review error logs
- [ ] Run full test suite
- [ ] Check git status

---

## Production Deployment Checklist

Before deploying to production:
- [ ] All critical tests pass
- [ ] No temporal leakage detected
- [ ] Edge calculation verified
- [ ] Data quality validated
- [ ] Coverage > 60%
- [ ] Documentation updated
- [ ] Production runbook complete
- [ ] Manual testing on realistic data

---

**Remember**: These standards prevent losing money. Take temporal leakage and betting logic seriously!
