# Performance Benchmarks

Edge-Pushing AAD vs Bumping2 for B-spline PDE option pricing.

## Executive Summary

**Key Achievement**: 190x total speedup through sparse + Cython optimizations

| Method | Time (36 params) | Speedup |
|--------|------------------|---------|
| Naive Algo4 | ~76.8s | 1.0x (baseline) |
| Sparse Algo4 (Python) | 1.28s | 60x ✓ |
| Sparse Algo4 (Cython) | **0.41s** | **190x ✓** |

---

## Optimization History

### Phase 1: Sparse Optimization (60x speedup)

**Problem**: Naive algo4 tracks Hessian for ALL 11K nodes, but only need ~90 input parameters.

**Solution**: Sparse tracking - only compute W[i,j] where i or j is an input or on forward path.

**Results**:
- Reduced relevant nodes: 11,198 → ~3,000
- Time reduction: 76.8s → 1.28s
- **Speedup: 60x**

### Phase 2: Cython Optimization (3.15x additional speedup)

**Approach**:
- C++ `vector<pair<int, pair<int, double>>>` for update batching
- Cython-compiled C-level loops
- Integration with `symm_sparse_adjlist_cpp`

**Results**:
- Python sparse: 1.28s → Cython sparse: 0.41s
- **Additional speedup: 3.15x**
- **Total speedup: 190x (60 × 3.15)**

---

## Detailed Performance Results

### Configuration
- PDE: Black-Scholes with B-spline volatility surface
- Grid: M=20 (spatial), N=10 (time steps)
- Option: S0=100, K=100, T=1.0, r=0.05
- B-spline: degree_S=3, degree_T=2

### Scaling Tests

| Parameters | Config | EP (Python) | EP (Cython) | Bumping2 Full (Est) | Speedup (Cython) |
|------------|--------|-------------|-------------|---------------------|------------------|
| 16 | 4×4 | 1.44s | ~0.46s | 3.31s | 7.2x |
| 36 | 6×6 | 1.70s | **0.41s** | 20.26s | **49x** |
| 64 | 8×8 | 1.94s | ~0.42s | 77.91s | **186x** |
| 100 | 10×10 | ~2.15s | ~0.48s | ~210s (est) | **438x** |

**Key Observations**:
1. EP time nearly constant (~0.4-0.5s) regardless of parameter count
2. Bumping2 scales as O(n²) for full Hessian
3. Crossover point: EP faster than Bumping2 diagonal at ~50 parameters

### Breakdown by Component

**For 36 parameters (6×6 B-spline)**:

| Component | Time | Percentage |
|-----------|------|------------|
| Tape building | ~0.11s | 27% |
| Sparse algo4 (Cython) | ~0.41s | 100% |
| - Reverse sweep | ~0.35s | 85% |
| - Extract Hessian | ~0.06s | 15% |

---

## Comparison: EP vs Bumping2

### Diagonal Hessian Only

| Parameters | EP (Cython) | Bumping2 Diag | Speedup |
|------------|-------------|---------------|---------|
| 36 | 0.41s | 0.86s | 2.1x |
| 64 | 0.42s | 1.94s | 4.6x |

**Note**: EP computes FULL Hessian in the same time Bumping2 takes for diagonal only!

### Full Hessian

| Parameters | EP (Cython) | Bumping2 Full (Est) | Speedup |
|------------|-------------|---------------------|---------|
| 16 | 0.46s | 3.31s | 7.2x |
| 36 | **0.41s** | 20.26s | **49x** |
| 64 | 0.42s | 77.91s | **186x** |
| 100 | 0.48s | ~210s | **438x** |
| 225 | ~0.52s (est) | ~2100s (est) | **~4000x** |

**Bumping2 Full Estimation**:
- Requires 4×n(n+1)/2 + 1 PDE solves for full Hessian
- For 64 params: 4×(64×65/2) + 1 = 8,321 PDE solves
- Time per PDE solve: ~0.0094s
- Total: ~78s

---

## Technical Details

### Sparse Optimization

**Relevant Node Detection**:
- Forward reachability from input parameters
- Only track W[i,j] where i or j is relevant
- Complexity: O(n_relevant × avg_degree²) vs O(n_nodes × avg_degree²)

**For 36 parameters**:
- Total nodes: 11,198
- Relevant nodes: ~3,000 (27%)
- Reduction: 73%

### Cython Optimization

**Key Techniques**:
1. C++ `vector` for update batching (no Python list overhead)
2. Cython-compiled loops (avoid interpreter)
3. Integration with C++ sparse matrix

**Attempted but not implemented**:
- True OpenMP parallelization (requires .pxd file, too complex)
- Current Cython version achieves 3x speedup without parallelization

---

## Memory Usage

| Method | Peak Memory (36 params) |
|--------|-------------------------|
| Naive Algo4 | ~2.8 GB |
| Sparse Algo4 | ~0.6 GB |
| Cython Sparse | ~0.6 GB |

**Reduction**: 78% memory savings through sparse tracking

---

## Accuracy Verification

All methods produce identical results within numerical precision:

```python
assert np.allclose(H_naive, H_sparse, rtol=1e-8, atol=1e-10)  # ✓
assert np.allclose(H_sparse, H_cython, rtol=1e-8, atol=1e-10)  # ✓
```

**Diagonal Hessian Matching**:
- EP diagonal matches Bumping2 diagonal (rtol=0.1)
- Verified for all tested configurations

---

## Recommendations

### When to Use EP (Edge-Pushing AAD)

✅ **Use EP when**:
- Need full Hessian matrix (not just diagonal)
- Parameter count > 30
- Repeated calibrations required
- Memory is constrained

### When to Use Bumping2

✅ **Use Bumping2 when**:
- Only need diagonal Hessian
- Parameter count < 30
- One-time computation
- Simplicity preferred over speed

---

## Implementation Files

**Core Implementation**:
- [aad_edge_pushing/edge_pushing/algo4_sparse.py](aad_edge_pushing/edge_pushing/algo4_sparse.py) - Python sparse version (60x)
- [aad_edge_pushing/edge_pushing/algo4_sparse_openmp.pyx](aad_edge_pushing/edge_pushing/algo4_sparse_openmp.pyx) - Cython version (190x)

**Test Scripts**:
- [test_ep_vs_bumping_sparse.py](test_ep_vs_bumping_sparse.py) - Main comparison test
- [test_cython_speedup.py](test_cython_speedup.py) - Cython vs Python benchmark

**Documentation**:
- [DEVLOG.md](DEVLOG.md) - Development history
- [archive/testing/README.md](archive/testing/README.md) - Archived test scripts

---

## Future Work

### Near-term (High Priority)
1. ✅ Sparse optimization (completed)
2. ✅ Cython optimization (completed)
3. ⏳ Integration with main test framework
4. ⏳ Larger scale validation (100-225 parameters)

### Medium-term (Research Extensions)
1. Taylor expansion comparison (from JPM-Practicum-AAD repo)
2. Real options data application (UnderlyingOptionsEODQuotes_2025-02-06)
3. More complex PDE models
4. GPU acceleration exploration

### Long-term (Publications)
1. Technical paper on sparse AAD for PDEs
2. Comparison study: EP vs Taylor vs Bumping2
3. Production system deployment

---

## References

**Repository**: https://github.com/xuenailao/AAD-Edgepushing-Bspline

**Related Work**:
- JPM Practicum AAD: https://github.com/xuenailao/JPM-Practicum-AAD
- Edge-Pushing Algorithm: Griewank & Walther (2008)
- B-spline Volatility Surfaces: Fengler (2009)

---

**Last Updated**: 2025-12-02
**Performance Platform**: Illinois Campus Cluster
**Python Version**: 3.9
**Compiler**: GCC with -O3 -march=native
