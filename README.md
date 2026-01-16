# Continuous Jump Model (CJM) for VNINDEX

## Project Overview
Implementation of a Continuous Jump Model (CJM) with mode loss for VNINDEX daily data, including a simulation study for optimal jump-penalty λ selection following the BAC vs λ methodology.

## Dataset
- **Date Range**: 2018-01-01 to 2025-12-31 (filtered to available data)
- **Source**: VNINDEX daily stock market data
- **Rows**: ~1,800+ trading days after filtering

## Selected Features
Based on technical analysis and market breadth indicators:

1. **ADX** - Average Directional Index (trend strength)
2. **URSI** - Ultimate RSI (momentum oscillator)
3. **BBWP** - Bollinger Band Width Percentile
4. **BB_PCTB** - Bollinger Band %B
5. **URSI_0_50** - Percentage of stocks with URSI between 0-50
6. **URSI_0_20** - Percentage of stocks with URSI between 0-20

## Methodology

### Step 1: HMM Fitting
- Fit Gaussian HMM with K=2 states on log returns
- Extract parameters: μ, σ, transition matrix P, stationary distribution π

### Step 2: Simulation
- Generate N_sim=1024 sequences of length T=1000
- Each sequence follows the fitted HMM dynamics

### Step 3: Lambda Scan
- Test λ values: logspace(-2, 5, 29)
- Models tested:
  - **JM**: Discrete Jump Model
  - **CJM_mode**: Continuous Jump Model with mode loss
- Metric: Balanced Accuracy with permutation fix

### Step 4: Real Data Application
- Apply best λ to real VNINDEX data with selected features
- Standardize features using StandardScaler
- Fit CJM_mode model and extract regime labels

## Results

### Optimal Lambda Selection
(Results will be populated after running the simulation)
- **JM Best Lambda**: TBD
- **CJM_mode Best Lambda**: TBD
- **Mean BAC Achieved**: TBD

### Regime Characteristics
(To be updated after model fitting)
- Regime 0: Bull/Bear characteristics
- Regime 1: Bull/Bear characteristics

## File Structure
```
jump_lambda_vn/
├── configs/
│   └── daily.yaml              # Configuration file
├── data/
│   └── vnindex_full.csv        # Input data with features
├── src/
│   ├── step0_prepare_vnindex.py
│   ├── step1_fit_hmm.py
│   ├── step2_simulate_hmm.py
│   ├── step3_lambda_scan.py
│   ├── step4_fit_jumpmodel_realdata.py
│   ├── metrics.py
│   └── utils_io.py
├── outputs/
│   ├── step1_params/
│   │   └── hmm_K2.json
│   ├── step2_sim/
│   │   └── K2/                 # 1024 simulation files
│   ├── step3_lambda/
│   │   ├── lambda_scores.csv
│   │   ├── best_lambda.json
│   │   └── bac_vs_lambda.png
│   └── step4_apply/
│       └── regimes_daily.csv
└── README.md
```

## Requirements
```
numpy
pandas
scipy
scikit-learn
hmmlearn
jumpmodels
matplotlib
tqdm
pyyaml
```

## Usage
```bash
# Run all steps sequentially
python run_all.py

# Or run individual steps
cd src
python step0_prepare_vnindex.py
python step1_fit_hmm.py
python step2_simulate_hmm.py
python step3_lambda_scan.py
python step4_fit_jumpmodel_realdata.py
```

## Key Outputs
1. **lambda_scores.csv**: BAC scores for all λ values tested
2. **best_lambda.json**: Optimal λ for each model type
3. **bac_vs_lambda.png**: Visualization of BAC vs λ
4. **regimes_daily.csv**: Daily regime labels and probabilities

## Notes
- All model fitting is deterministic given the seed (123)
- Missing values in features are forward-filled then dropped
- Degenerate simulations (all one regime) are kept as per methodology
- Label switching is handled via permutation-invariant BAC