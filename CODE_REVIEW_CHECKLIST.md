# Code Review Checklist

Use this checklist for self-reviews (solo dev) or peer reviews (2-person team).

---

## Critical Issues (Must Fix Before Merge)

### Temporal Leakage Prevention
- [ ] No features use future data (all lag/rolling features use `.shift()`)
- [ ] Walk-forward validation uses strict chronological splits
- [ ] Training data dates < validation data dates < test data dates
- [ ] No "look-ahead" bias in feature engineering
- [ ] Feature creation order is documented and validated

### Data Quality
- [ ] No duplicate (PLAYER_ID, GAME_DATE) combinations
- [ ] All DataFrames sorted by (PLAYER_ID, GAME_DATE) where required
- [ ] No unexpected missing values in critical columns
- [ ] Data types are appropriate (int32 vs int64, float32 vs float64)
- [ ] Date columns are datetime objects, not strings

### Betting Logic
- [ ] Edge calculation is mathematically correct
- [ ] Kelly sizing produces reasonable bet sizes (< 10% of bankroll)
- [ ] Negative expected value bets are filtered out
- [ ] Risk limits are enforced (max bet, max daily exposure)
- [ ] No betting on props with stale odds (> 10 minutes old)

---

## Important Issues (Should Fix)

### Error Handling
- [ ] Custom exceptions used for domain-specific errors
- [ ] Critical operations have try-except blocks
- [ ] Errors include helpful context (not just "Error occurred")
- [ ] Graceful degradation for non-critical failures
- [ ] Resource cleanup in finally blocks or context managers

### Code Quality
- [ ] Type hints on all public functions
- [ ] Docstrings for complex functions (Google style)
- [ ] No `print()` statements (use `logging` instead)
- [ ] No hardcoded paths (use config or Path objects)
- [ ] No magic numbers (use named constants)
- [ ] Variable names are clear and descriptive

### Testing
- [ ] New features have unit tests
- [ ] Critical paths have integration tests
- [ ] Tests are fast (< 1 second for unit tests)
- [ ] Tests use fixtures, not production data
- [ ] No tests are skipped or marked as xfail without reason

### Configuration
- [ ] No credentials in code (use environment variables)
- [ ] Configuration values are externalized (not hardcoded)
- [ ] Paths are constructed with Path objects
- [ ] Default values are reasonable

---

## Nice-to-Have (Optional Improvements)

### Documentation
- [ ] Complex logic has inline comments explaining "why"
- [ ] New features documented in Feature Registry
- [ ] API changes documented
- [ ] README updated if public interface changed

### Performance
- [ ] No obvious performance issues (nested loops on large data)
- [ ] Vectorized operations used where possible
- [ ] No unnecessary data copies
- [ ] Memory usage is reasonable

### Code Style
- [ ] Code formatted with black
- [ ] Imports sorted with isort
- [ ] No flake8 warnings (except unavoidable ones with # noqa)
- [ ] Line length < 100 characters

---

## Domain-Specific Checks

### Feature Engineering
- [ ] Feature has clear business logic (documented)
- [ ] Feature correlation with PRA is measured
- [ ] Feature doesn't cause multicollinearity (VIF < 10)
- [ ] Missing values handled appropriately (imputation or default)
- [ ] Feature scales appropriately (normalization if needed)

### Model Training
- [ ] Model hyperparameters documented
- [ ] Cross-validation strategy is time-aware (no random splits)
- [ ] Feature importance extracted and analyzed
- [ ] Model serialization/deserialization tested
- [ ] Overfitting checked (train vs validation performance)

### Data Pipeline
- [ ] Pipeline stages are clearly separated
- [ ] Intermediate outputs can be cached
- [ ] Pipeline can resume from failure point
- [ ] Progress logging for long operations
- [ ] Data validation at pipeline entry/exit points

---

## Before Merging to Main

### Automated Checks
- [ ] `pytest tests/` passes
- [ ] `black src/` runs without changes
- [ ] `flake8 src/` shows no errors
- [ ] `mypy src/betting src/evaluation` passes (if strict mode enabled)

### Manual Verification
- [ ] Code runs on small test dataset without errors
- [ ] Changes don't break existing functionality (regression test)
- [ ] Performance is acceptable (build time < 5 minutes for full dataset)
- [ ] Documentation updated if needed

### Team Coordination (2-person team)
- [ ] Changes communicated to team member
- [ ] Breaking changes flagged
- [ ] Migration path documented for breaking changes

---

## Review Outcomes

**Option 1: Approve**
- All critical issues resolved
- Important issues resolved or have accepted trade-offs
- Code is ready to merge

**Option 2: Request Changes**
- Critical issues found that must be fixed
- List specific issues in review comments

**Option 3: Comment Only**
- No blocking issues
- Suggestions for future improvements
- Code can merge as-is

---

## Self-Review Workflow (Solo Developer)

1. **Immediately After Coding (5 minutes)**
   - Run automated checks (black, flake8, pytest)
   - Fix any obvious errors
   - Commit to feature branch

2. **Before Push (15 minutes)**
   - Review all critical checks above
   - Run full test suite
   - Test on small dataset end-to-end
   - Push to feature branch

3. **Before Merge to Dev (30 minutes)**
   - Print out the code diff
   - Read through line-by-line with checklist
   - Test on full dataset (if performance-critical change)
   - Merge to dev

4. **Before Merge to Main (1 hour)**
   - Run complete test suite including integration tests
   - Manual validation on realistic data
   - Update documentation
   - Merge to main

---

## Peer Review Workflow (2-person Team)

1. **Author Preparation**
   - Complete all automated checks
   - Add checklist to PR description
   - Self-review using checklist
   - Tag reviewer

2. **Reviewer Process**
   - Check out branch and run code locally
   - Go through checklist systematically
   - Add comments for any issues
   - Approve or request changes

3. **Author Response**
   - Address all review comments
   - Push fixes
   - Re-request review if needed

4. **Final Approval**
   - Reviewer approves
   - Author merges to main
   - Author deletes feature branch

---

## Common Issues to Watch For

### Temporal Leakage (Most Critical)
```python
# BAD: Rolling average includes current game
df['PRA_L5_mean'] = df.groupby('PLAYER_ID')['PRA'].rolling(5).mean()

# GOOD: Rolling average excludes current game
df['PRA_L5_mean'] = (
    df.groupby('PLAYER_ID')['PRA']
    .shift(1)  # Exclude current game
    .rolling(5, min_periods=1)
    .mean()
)
```

### Duplicate Data
```python
# BAD: Merging without deduplication
result = game_logs.merge(ctg_stats, on=['Player', 'Season'])

# GOOD: Deduplicate before merging
ctg_dedup = ctg_stats.drop_duplicates(subset=['Player', 'Season'], keep='first')
result = game_logs.merge(ctg_dedup, on=['Player', 'Season'])
```

### Error Handling
```python
# BAD: Silent failure
try:
    stats = load_ctg_stats(player)
except:
    stats = None

# GOOD: Specific exception with logging
try:
    stats = load_ctg_stats(player)
except FileNotFoundError:
    logger.warning(f"CTG stats not found for {player}, using league averages")
    stats = DEFAULT_LEAGUE_AVERAGES
except Exception as e:
    logger.error(f"Unexpected error loading CTG stats for {player}: {e}")
    raise
```

### Magic Numbers
```python
# BAD: Unexplained magic numbers
if edge > 0.03 and odds > 1.5:
    bet_size = bankroll * 0.02

# GOOD: Named constants with rationale
MIN_EDGE = 0.03  # Minimum edge to overcome vig and variance
MIN_ODDS = 1.5   # Avoid heavy favorites (low edge, high variance)
KELLY_FRACTION = 0.25  # Use 1/4 Kelly to reduce variance

if edge > MIN_EDGE and odds > MIN_ODDS:
    kelly_full = (edge * odds - 1) / (odds - 1)
    bet_size = bankroll * kelly_full * KELLY_FRACTION
```

---

## Quick Reference: Review Time Estimates

| Change Type | Critical Checks | Important Checks | Total Review Time |
|-------------|-----------------|------------------|-------------------|
| Bug fix (< 50 lines) | 5 min | 5 min | 10 min |
| Small feature (< 200 lines) | 10 min | 10 min | 20 min |
| Medium feature (< 500 lines) | 20 min | 15 min | 35 min |
| Large feature (> 500 lines) | 30 min | 30 min | 60 min |
| Architecture change | 45 min | 30 min | 75 min |

---

## Appendix: Critical Code Patterns

### Pattern 1: Safe Merging
```python
# Always deduplicate before merging
df1 = df1.drop_duplicates(subset=merge_keys, keep='first')
df2 = df2.drop_duplicates(subset=merge_keys, keep='first')

# Merge with validation
before = len(df1)
result = df1.merge(df2, on=merge_keys, how='left', validate='1:1')
after = len(result)

if before != after:
    raise DataQualityError(f"Merge changed row count: {before} -> {after}")
```

### Pattern 2: Safe Feature Creation
```python
# Always exclude current game for temporal features
def create_temporal_feature(df: pd.DataFrame, stat: str, window: int) -> pd.Series:
    """
    Create temporal feature with proper leakage prevention.

    CRITICAL: Uses .shift(1) to exclude current game.
    """
    return (
        df.groupby('PLAYER_ID')[stat]
        .shift(1)  # CRITICAL: Exclude current game
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
```

### Pattern 3: Safe Train/Test Split
```python
# Always use chronological splits
def create_time_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time-based train/test split with validation."""
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)

    # Validate no overlap
    if train_end_dt >= test_start_dt:
        raise TemporalLeakageError(
            f"Training ends ({train_end_dt}) on or after test starts ({test_start_dt})"
        )

    train = df[df['GAME_DATE'] <= train_end_dt].copy()
    test = df[df['GAME_DATE'] >= test_start_dt].copy()

    # Validate no data overlap
    overlap = set(train.index) & set(test.index)
    if overlap:
        raise TemporalLeakageError(f"Found {len(overlap)} overlapping rows")

    return train, test
```
