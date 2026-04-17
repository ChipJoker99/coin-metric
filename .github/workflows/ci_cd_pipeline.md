# CI/CD Pipeline Evolution Workflow

## Goal

Define how the CI pipeline should evolve together with the project.

---

# Core Principle

Every new feature must include:
- tests
- CI integration
- regression safety (if ML)

---

# New modules

When adding a module in `src/`:

### Mandatory:
- unit tests in `tests/unit/`
- integration tests if it impacts the pipeline

---

# ML components

If you add:
- model
- loss function
- embedding logic

### Mandatory:
- `tests/ml_sanity/`
- `tests/regression/`

---

# Metrics rules

If the following change:

- Top-1 accuracy
- Top-5 accuracy
- retrieval performance

### CI must:
- fail if performance degrades beyond threshold
- block merge

---

# Dataset changes

If the dataset changes:
- update Golden Dataset
- rerun regression suite

---

# CI update rule

If you add a new test category:

- update `.github/workflows/ci.yml`

---

# Order of execution (MANDATORY)

1. unit tests
2. integration tests
3. retrieval tests
4. ML sanity tests
5. regression tests
6. performance tests

---

# Forbidden

- bypassing CI
- disabling tests for speed
- mixing training inside CI

---

# Success criteria

CI is valid only if it is:
- deterministic
- reproducible
- regression-free