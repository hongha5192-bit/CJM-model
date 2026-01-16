# CJM K=3 Fast Mode Implementation

## Overview
This implementation provides a K=3 (three-regime) Continuous Jump Model (CJM) for VNINDEX daily data (2018-2025), optimized for Mac Air M4 with Fast Mode settings.

## Fast Mode Optimizations
- **Simulations**: 256 sequences (reduced from 1024)
- **Lambda grid**: 21 points (reduced from 29)
- **Parallel workers**: Capped at 4
- **Grid size**: 0.05 (optimal for K=3)
- **Iterations**: Reduced n_init and max_iter for faster convergence

## Quick Start

### Run Complete Pipeline
```bash
python run_K3_fast.py
```

This will execute all steps sequentially:
1. Data preparation (if needed)
2. Fit HMM with K=3
3. Generate 256 simulations
4. Lambda scan (21 points)
5. Apply CJM to real data

### Run Individual Steps
```bash
cd src

# Step 1: Fit HMM K=3
python step1_fit_hmm_K3.py

# Step 2: Generate simulations
python step2_simulate_hmm_K3.py

# Step 3: Lambda scan
python step3_lambda_scan_K3.py

# Step 4: Apply to real data
python step4_fit_cjm_realdata_K3.py
```

## Configuration
Edit `configs/daily_K3.yaml` to adjust:
- Number of simulations (N_sim)
- Lambda grid points (n_grid)
- CJM optimization settings
- Features to use

## Output Files

### Step 1: HMM Parameters
- `outputs/step1_params/hmm_K3.json` - Fitted 3-state HMM parameters

### Step 2: Simulations
- `outputs/step2_sim/K3/` - 256 simulated sequences
- `outputs/step2_sim/K3/simulation_summary_K3.json` - Simulation statistics

### Step 3: Lambda Selection
- `outputs/step3_lambda/lambda_scores_K3.csv` - BAC scores for each lambda
- `outputs/step3_lambda/best_lambda_K3.json` - Selected optimal lambda
- `outputs/step3_lambda/bac_vs_lambda_K3.png` - Visualization

### Step 4: Real Data Results
- `outputs/step4_apply/regimes_daily_K3.csv` - Daily regime predictions
- `outputs/step4_apply/cjm_model_params_K3.json` - Model parameters
- `outputs/step4_apply/regime_analysis_K3.png` - Analysis visualization

## Features Used
- ADX (Average Directional Index)
- URSI (Ultimate RSI)
- BBWP (Bollinger Band Width Percentile)
- BB_PCTB (Bollinger Band %B)
- URSI_0_50 (Market breadth 0-50)
- URSI_0_20 (Market breadth 0-20)

## K=3 Regime Interpretation
The three regimes typically represent:
- **State 0**: Low volatility/calm market
- **State 1**: Medium volatility/transitional
- **State 2**: High volatility/stressed market

## Performance Notes
- Complete pipeline runs in ~10-15 minutes on Mac Air M4
- Memory usage stays under 8GB
- Can be interrupted and resumed (lambda scan saves incrementally)

## Troubleshooting
- If memory issues occur, reduce N_sim in config
- Lambda scan can be resumed from partial results
- Delete output files to force re-computation of specific steps