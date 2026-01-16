# K=3 CJM Implementation - Results Summary

## Fixes Applied (from Review.MD)

### Step 1: HMM Fitting (`step1_fit_hmm_K3.py`)
✅ **Fixed `init_params`**: Changed from invalid `'kmeans++'` to valid `'stmc'`
✅ **Fixed stationary distribution**: Now correctly computed as left eigenvector of P with eigenvalue 1
✅ **Fixed log-likelihood**: Removed erroneous multiplication by len(X)

### Step 2: Simulations (`step2_simulate_hmm_K3.py`)
✅ **Fixed switch counting**: Subtracted 1 to exclude first row NaN from diff()

## Corrected Results

### HMM K=3 Parameters (Corrected)
```
Number of states: 3
Training period: 2018-01-03 to 2025-11-24
Number of observations: 1971
Log-likelihood: 6213.98 (corrected - not inflated)

State parameters (sorted by volatility):
  State 0: μ= 0.0017, σ= 0.0077  (Low volatility)
  State 1: μ=-0.0029, σ= 0.0208  (Medium volatility)
  State 2: μ=-0.0058, σ= 0.0239  (High volatility)

Stationary distribution (corrected - true stationary):
  π[0] = 0.7740  (77.4% in low volatility)
  π[1] = 0.0780  (7.8% in medium volatility)
  π[2] = 0.1481  (14.8% in high volatility)

Transition matrix:
  P[0→*] = [0.9294, 0.0605, 0.0101]  (Low vol is persistent)
  P[1→*] = [0.6992, 0.0481, 0.2527]  (Medium vol transitions quickly)
  P[2→*] = [0.0007, 0.1853, 0.8141]  (High vol is persistent)
```

### Simulation Validation (256 simulations)
```
State Frequencies (mean ± std):
  State 0: 0.772 ± 0.041 (expected: 0.774) ✅
  State 1: 0.078 ± 0.009 (expected: 0.078) ✅
  State 2: 0.149 ± 0.036 (expected: 0.148) ✅

Mean switches per 1000 observations: 157.0 (corrected)
```

## Key Insights

1. **Market Structure**: VNINDEX spends most time (77.4%) in low volatility regime, consistent with normal market conditions

2. **Regime Persistence**:
   - Low volatility (State 0): 92.94% chance of staying → average duration ~14 days
   - High volatility (State 2): 81.41% chance of staying → average duration ~5.4 days
   - Medium volatility (State 1): Only 4.81% chance of staying → transitional state

3. **Asymmetric Transitions**:
   - Low → High: Only 1.01% direct transition (rare crisis events)
   - High → Low: Only 0.07% direct transition (crises don't end abruptly)
   - Most transitions go through medium volatility state

4. **Simulation Quality**: The simulations accurately reproduce the stationary distribution, confirming the HMM fitting and simulation procedures are working correctly

## Files Generated

- `outputs/step1_params/hmm_K3.json` - Corrected HMM parameters with proper stationary distribution
- `outputs/step2_sim/K3/*.parquet` - 256 simulation files with corrected state generation
- `outputs/step2_sim/K3/simulation_summary_K3.json` - Validation statistics

## Lambda Selection Status

The lambda scan was attempted but requires significant computation time due to K=3 complexity. For production use, consider:
1. Using parallel processing with more workers
2. Reducing grid_size from 0.05 to 0.1 for faster convergence
3. Using a subset of simulations (e.g., 50-100) for initial lambda selection

## Recommendations

1. **For Lambda Selection**: Run overnight with full 256 simulations or use high-performance computing
2. **For Real-time Applications**: Consider K=2 for faster inference
3. **For Research**: The corrected K=3 model provides valuable insights into VNINDEX regime dynamics