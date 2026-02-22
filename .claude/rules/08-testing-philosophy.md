# Testing Philosophy

> **PORTABILITY NOTICE**: This file is designed to be **reusable across any Python project**.

## Core Principle

**Tests exist to challenge and verify code, not just to pass.** A passing test suite with weak assertions provides false confidence. Tests should catch bugs, not hide them.

## Anti-Patterns to Avoid

### 1. Vacuous Assertions

**BAD:**
```python
def test_pipeline_runs():
    result = run_pipeline()
    assert True  # Always passes
    assert isinstance(result, dict)  # Only checks type, not content
```

**GOOD:**
```python
def test_pipeline_produces_valid_output():
    result = run_pipeline()
    assert "weights" in result
    assert np.allclose(result["weights"].sum(), 1.0, atol=1e-10)
    assert result["entropy"] > 0
```

### 2. Overly Permissive Tolerances

**BAD:**
```python
# Z-scored data should have std ≈ 1.0
assert abs(data.std() - 1.0) < 0.5  # 50% tolerance!

# ERC risk contributions should be equal
assert np.allclose(rc, expected_rc, atol=0.003)  # 0.3% absolute on ~2% values = 15% relative
```

**GOOD:**
```python
# Z-scored data: tight tolerance for deterministic computation
assert abs(data.std() - 1.0) < 1e-6

# ERC: tight tolerance matching solver precision
assert np.allclose(rc, expected_rc, atol=1e-5)
```

**Rule of thumb:** Use tolerances appropriate to the computation:
- Deterministic formulas: `atol=1e-10` to `1e-12`
- Iterative solvers: `atol=1e-6` to `1e-8`
- Statistical estimates: `atol=1e-4` to `1e-6`
- Never exceed `1e-2` (1%) without explicit justification in comments

### 3. Silent Skips That Mask Failures

**BAD:**
```python
def test_window_quality(windows):
    if windows.shape[0] == 0:
        pytest.skip("No windows")  # Hides generation failure
    assert not torch.isnan(windows).any()
```

**GOOD:**
```python
def test_window_quality(windows):
    assert windows.shape[0] > 0, "Fixture must generate windows"
    assert not torch.isnan(windows).any()
```

**Rule:** Use `pytest.skip()` only for environment-dependent tests (OS, GPU). Never skip because input data is bad—that's a test failure.

### 4. Conditional Assertions

**BAD:**
```python
if result.get("action"):
    assert "fix" in result["action"]  # Passes when action is None
```

**GOOD:**
```python
assert result.get("action") is not None, "Action required for this case"
assert "fix" in result["action"]
```

### 5. Testing Only Happy Path

**BAD:**
```python
def test_solver():
    w = optimize(Sigma)
    assert w.shape == (n,)
    assert np.isfinite(w).all()
    # Never checks if solver actually improved objective!
```

**GOOD:**
```python
def test_solver_improves_objective():
    w_init = np.ones(n) / n
    w_opt = optimize(Sigma)

    # Verify improvement
    obj_init = compute_objective(w_init, Sigma)
    obj_opt = compute_objective(w_opt, Sigma)
    assert obj_opt < obj_init, "Solver must improve objective"

    # Verify convergence (gradient small)
    grad = compute_gradient(w_opt, Sigma)
    assert np.linalg.norm(grad[w_opt > 1e-6]) < 1e-6
```

### 6. xfail as Excuse

**BAD:**
```python
if AU < 2:
    pytest.xfail("Model didn't learn enough")  # Hides training bug
```

**GOOD:**
```python
assert AU >= 2, (
    f"Model must learn at least 2 active units for this test. "
    f"Got AU={AU}. Check training hyperparameters."
)
```

### 7. Trivial Synthetic Data

**BAD:**
```python
# Pure Gaussian white noise - no structure to test
returns = np.random.randn(n_days, n_stocks) * 0.01
```

**GOOD:**
```python
# Factor structure + autocorrelation + heteroskedasticity
B_true = np.random.randn(n_stocks, n_factors) * 0.3
z = np.zeros((n_days, n_factors))
for t in range(1, n_days):
    z[t] = 0.3 * z[t-1] + np.random.randn(n_factors) * 0.01  # AR(1)
returns = z @ B_true.T + np.random.randn(n_days, n_stocks) * 0.005
```

### 8. CLI Tests Without Output Verification

**BAD:**
```python
result = subprocess.run(["python", "script.py"])
assert result.returncode == 0  # Only checks exit code
```

**GOOD:**
```python
result = subprocess.run(["python", "script.py"], capture_output=True)
assert result.returncode == 0, f"Failed: {result.stderr}"

# Verify outputs exist and are valid
assert Path("output.csv").exists()
df = pd.read_csv("output.csv")
assert len(df) > 0
assert "expected_column" in df.columns
```

---

## Best Practices

### Formula Verification
For mathematical formulas, verify with known inputs:
```python
def test_entropy_formula():
    # Equal contributions: H = ln(n)
    n = 10
    contributions = np.ones(n) / n
    H = compute_entropy(contributions)
    assert np.isclose(H, np.log(n), atol=1e-12)
```

### Edge Case Coverage
Always test boundaries:
```python
def test_edge_cases():
    # Single element
    assert compute_entropy(np.array([1.0])) == 0.0

    # Near-zero element
    w = np.array([0.99, 0.01])
    H = compute_entropy(w)
    assert 0 < H < np.log(2)

    # All zeros except one
    w = np.array([1.0, 0.0, 0.0])
    assert compute_entropy(w) == 0.0
```

### Fixture Validation
Validate fixtures produce expected properties:
```python
@pytest.fixture
def ar1_returns():
    rho = 0.3
    returns = generate_ar1(rho=rho, n=1000)
    # Validate fixture before using
    actual_rho = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    assert 0.25 < actual_rho < 0.35, f"Fixture rho={actual_rho}, expected ~{rho}"
    return returns
```

### Intermediate Assertions in Multi-Stage Tests
```python
def test_full_pipeline():
    # Stage 1
    windows = create_windows(data)
    assert windows.shape[0] > 0, "Stage 1 failed: no windows"
    assert not torch.isnan(windows).any(), "Stage 1: NaN in windows"

    # Stage 2
    model = train_vae(windows)
    assert model.encoder is not None, "Stage 2 failed: no encoder"

    # Stage 3
    latents = model.encode(windows)
    assert torch.isfinite(latents).all(), "Stage 3: non-finite latents"
```

---

## Test Evolution

Tests should evolve with the code:

1. **When fixing a bug:** Add a test that would have caught it
2. **When changing behavior:** Update tests to reflect new expected behavior
3. **When relaxing constraints:** Document why in the test
4. **When tightening constraints:** Verify all edge cases still pass

**Never weaken a test just to make it pass.** If a test fails, either:
- Fix the code (if the test is correct)
- Fix the test (if the code is correct and test was wrong)
- Document why the tolerance must be relaxed (if both are correct but precision is limited)

---

## Checklist Before Committing Tests

- [ ] All assertions verify **values**, not just types/shapes
- [ ] Tolerances are justified and as tight as possible
- [ ] No `assert True` or `assert isinstance()` without value checks
- [ ] No `pytest.skip()` hiding data/logic failures
- [ ] Edge cases covered (empty, single element, boundary values)
- [ ] Synthetic data has realistic structure if testing ML/statistics
- [ ] Multi-stage tests have intermediate assertions
- [ ] CLI/integration tests verify output content, not just exit codes
