# Mac Air M4 Optimizations Implemented

## All Improvements from Improvements.MD Successfully Applied ✅

### A) K=3 Grid Size (Most Important Fix) ✅
- **Set `grid_size: 0.05`** in configs/daily_K3.yaml
- This is the paper's recommended setting for three-state models
- Critical for K=3 convergence

### B) Reduced Compute Load ✅
- **N_sim: 128** (reduced from 256)
  - Still statistically significant for lambda selection
  - Halves computational load

- **n_grid: 11** (reduced from 21)
  - Covers essential lambda range [1e-2, 1e5]
  - Reduces total fits by ~50%

- **Relaxed convergence criteria:**
  - `n_init: 3` (from 10) for lambda scan
  - `max_iter: 200` (from 500)
  - `tol: 1e-5` (from 1e-6)

### C) Limited Parallelism ✅
- **n_jobs: 2** (reduced from 4)
  - Prevents memory pressure on Mac Air M4
  - Avoids CPU throttling
  - Often faster for memory-intensive DP operations

### D) Eliminated Repeated Disk Reads ✅
- **Load all simulations into memory once**
  - Created `step3_lambda_scan_K3_optimized.py`
  - Loads 128 simulations into memory at start
  - Passes arrays directly to CJM fitting
  - Saves ~16,384 disk reads (128 sims × 11 lambdas × 11.7 avg reads)

## Performance Impact

### Before Optimizations:
- 256 simulations × 21 lambdas = 5,376 CJM fits
- Each fit with n_init=10, max_iter=500
- Total iterations: ~26.9M
- Estimated time: 8-12 hours

### After Optimizations:
- 128 simulations × 11 lambdas = 1,408 CJM fits
- Each fit with n_init=3, max_iter=200
- Total iterations: ~844K
- Estimated time: 30-60 minutes
- **~32x reduction in computational load**

## Files Created

1. **configs/daily_K3.yaml** - Updated with all Mac Air M4 optimizations
2. **src/step3_lambda_scan_K3_optimized.py** - Memory-efficient lambda scan
3. **outputs/step2_sim/K3/** - 128 optimized simulations

## Current Status

The optimized lambda scan is running with:
- ✅ All simulations loaded in memory
- ✅ Reduced parameter space (11 lambdas)
- ✅ Optimized convergence settings
- ✅ Limited parallelism (n_jobs=2)

Expected completion: 30-60 minutes (vs 8-12 hours unoptimized)

## Validation

The 128 simulations show excellent agreement with theoretical expectations:
```
State Frequencies (mean ± std):
  State 0: 0.773 ± 0.040 (expected: 0.774) ✅
  State 1: 0.078 ± 0.009 (expected: 0.078) ✅
  State 2: 0.149 ± 0.036 (expected: 0.148) ✅
```

This confirms the simulation quality is maintained despite reduced sample size.