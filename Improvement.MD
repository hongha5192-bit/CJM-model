You are Claude acting as a senior quant engineer. Your task is to implement a Continuous Jump Model (CJM) regime-detection pipeline for VNINDEX daily data (2018–2025) with K=3 regimes, and to run exactly ONE simulation study to select the jump penalty hyperparameter λ, following the workflow and outputs defined in the provided MD spec.

Core references (must follow):
- CJM objective: hidden state is a probability vector on the simplex; penalize changes in consecutive probability vectors using squared ℓ1 norm; λ/4 scaling. Coordinate descent alternates between fitting model parameters and fitting the hidden state sequence; the hidden-state fitting can be approximated via DP on a discretized simplex. Mode loss is an extra penalty term that requires the DP approximation. (Do not re-derive; implement using the `jumpmodels` library.)
- Simulation study for λ: simulate many sequences of fixed length, fit CJM for each λ on each simulated sequence, evaluate recovery of true states using permutation-invariant Balanced Accuracy (BAC), and pick λ that maximizes mean BAC.
- For K=3 continuous models, discretize the simplex using a uniform grid size δ = 0.05 (compute reasons + paper guidance).

IMPORTANT CONSTRAINTS (Mac Air M4):
- Must be feasible on a laptop (limited CPU/RAM). Use small-but-reasonable defaults and implement a “FAST MODE” config that runs end-to-end in a practical time.
- Parallelism is allowed but must be conservative (cap workers to 4 by default). Avoid huge intermediate files. Write results incrementally so the run can resume.

PROJECT GOAL:
1) Implement λ-selection simulation study (ONE study only) for K=3.
2) Fit CJM (K=3) on real VNINDEX daily data (2018–2025) using the selected λ.
3) Output files MUST match names/structure in the MD spec:
   - outputs/step3_lambda/lambda_scores.csv
   - outputs/step3_lambda/best_lambda.json
   - outputs/step4_apply/regimes_daily.csv
   - outputs/step3_lambda/bac_vs_lambda.png (plot)

DATA ASSUMPTIONS (VNINDEX file):
- Input is a CSV with at least:
  - date (YYYY-MM-DD)
  - OHLCV columns (open, high, low, close, volume) or close at minimum
  - additional indicator columns already computed (ADX, URSI, market breadth, Bollinger features, etc.)
- DO NOT compute indicators. Only select and standardize existing feature columns.
- You MAY compute log returns from close if not already present.

IMPLEMENTATION STRATEGY (must follow MD pipeline but only K=3):
Create a repo structure like:
jump_lambda_vn/
  data/vnindex_daily.csv
  configs/daily_K3.yaml
  outputs/
    step1_params/hmm_K3.json
    step2_sim/K3/...
    step3_lambda/lambda_scores.csv
    step3_lambda/best_lambda.json
    step3_lambda/bac_vs_lambda.png
    step4_apply/regimes_daily.csv
  src/
    step0_prepare_data.py
    step1_fit_hmm_K3.py
    step2_simulate_hmm_K3.py
    step3_lambda_scan_K3.py
    step4_fit_cjm_realdata_K3.py
    metrics.py
    utils_io.py

You must implement runnable scripts (CLI style) + a single config YAML. Include logging + tqdm progress bars.

========================
MODEL DEFINITIONS (what you implement via libraries)
========================
Use Python packages:
- numpy, pandas, scipy, scikit-learn
- hmmlearn (GaussianHMM)
- jumpmodels (JumpModel)
- matplotlib
- joblib (optional for parallel)

CJM concept (no need to fully re-derive in code comments, but keep correct usage):
- Hidden state at each time t is s_t ∈ Δ^{K-1} (probability simplex).
- Objective includes:
  - data fit loss weighted by s_t
  - jump penalty proportional to (λ/4) * ||s_{t-1} - s_t||_1^2
- Estimation uses coordinate descent:
  1) fit model parameters Θ given S
  2) fit S given Θ (QP; approximated with DP on discretized simplex)
- Mode loss: additional penalty term that promotes sparse probability vectors and improves transitions; requires DP approximation.

In practice, do NOT implement coordinate descent manually. Use `jumpmodels.jump.JumpModel` with:
- cont=True
- mode_loss=True
- grid_size=0.05 for K=3
- jump_penalty=λ

========================
STEP 0 — PREPARE REAL DATA (src/step0_prepare_data.py)
========================
Input: data/vnindex_daily.csv
Tasks:
1) Parse date column, sort ascending, filter to [2018-01-01, 2025-12-31] (or max available date in 2025).
2) Ensure numeric columns are floats.
3) Compute log return ret = log(close_t / close_{t-1}) if not present.
4) Feature selection:
   - Define feature_cols in config (a list). These are already computed indicators (ADX, URSI, breadth, Bollinger, etc.).
   - Extract X = df[feature_cols].
   - Handle missing values: forward-fill then back-fill, then drop any remaining NaNs.
5) Standardize features:
   - Use StandardScaler fit on the same period you will train CJM on (2018–2025 in this project).
   - Save scaler params to outputs (optional).
Output:
- A cleaned parquet/csv in outputs/ (optional), but at minimum return arrays for later scripts:
  - dates, X (T×F), ret (T,)

========================
STEP 1 — FIT GAUSSIAN HMM (K=3) FOR SIMULATION PARAMS (src/step1_fit_hmm_K3.py)
========================
Purpose:
- Only to generate realistic simulated sequences for λ-selection.
- Fit HMM on 1D returns series (ret) using hmmlearn GaussianHMM with n_components=3.

Must do:
- Use Dirichlet prior for transition probabilities: transmat_prior = 1 + 1e-8 (numerical stability).
- Initialize with k-means++ style randomness (hmmlearn init + random_state); run multiple restarts and choose best log-likelihood if needed.

Export JSON:
outputs/step1_params/hmm_K3.json with:
{
  "K": 3,
  "mu": [...],
  "sigma": [...],
  "P": [[...],[...],[...]],
  "pi": [...],
  "train_start": "...",
  "train_end": "...",
  "n_obs": ...,
  "seed": ...
}

Sanity checks:
- rows of P sum to 1
- sigma_k > 0
- pi sums to 1, non-negative

========================
STEP 2 — SIMULATE FROM FITTED HMM (ONE SIM STUDY) (src/step2_simulate_hmm_K3.py)
========================
Simulation design (compute-aware but faithful):
- This is the ONLY simulation study you will run (for λ selection).
- Use K=3 parameters from hmm_K3.json.
- Simulate N_sim sequences, each of length T=1000 (fixed).
- Keep sequences even if they contain a single regime (do not filter them out).
- Save each simulation with true states.

FAST MODE defaults (laptop-safe):
- N_sim = 256 (configurable; allow 128 for quick debug)
- T = 1000
- seed0 = 1000

Simulation procedure for each sim:
1) s_1 ~ Categorical(pi)
2) s_t ~ Categorical(P[s_{t-1}, :])
3) y_t ~ Normal(mu[s_t], sigma[s_t]^2)

Save each simulation file under:
outputs/step2_sim/K3/sim_T1000_seed{XXXX}_id{YYYY}.parquet
Columns: t, y, s_true, sim_id, seed

========================
STEP 3 — LAMBDA SCAN FOR CJM (K=3) (src/step3_lambda_scan_K3.py)
========================
Goal:
- For each λ in a log-spaced grid:
  - Fit CJM (continuous jump model) with mode loss to each simulated sequence.
  - Predict regimes, compute permutation-invariant BAC vs s_true.
  - Aggregate mean/std BAC across simulations.
  - Pick λ* = argmax(mean_BAC).
- Save outputs EXACTLY like MD:
  - outputs/step3_lambda/lambda_scores.csv
  - outputs/step3_lambda/best_lambda.json
  - outputs/step3_lambda/bac_vs_lambda.png

Model to fit (jumpmodels):
from jumpmodels.jump import JumpModel
cjm = JumpModel(
  n_components=3,
  jump_penalty=lam,
  cont=True,
  mode_loss=True,
  grid_size=0.05,      # IMPORTANT for K=3
  n_init=FAST_N_INIT,
  max_iter=FAST_MAX_ITER,
  tol=FAST_TOL,
  random_state=seed,
  verbose=0
)

Input shape:
- X_sim must be (T, 1): X_sim = y.reshape(-1, 1)

Permutation-invariant BAC for K=3:
- Implement in src/metrics.py:
  - For all 6 permutations of labels {0,1,2}, map predicted labels -> permuted labels, compute balanced_accuracy_score, take max.
- Return BAC_best.

Lambda grid (FAST MODE):
- Use logspace from 1e-2 to 1e5 but fewer points:
  - n_grid = 21 (instead of 29)
  - np.logspace(-2, 5, 21)
Rationale:
- Still log-spaced like paper/MD; fewer points to fit laptop constraints.

Optimization settings (FAST MODE):
- n_init = 3 (not 10)
- max_iter = 300–500
- tol = 1e-6
- Workers: parallel over simulations for each λ, but cap n_jobs=4

Incremental saving / resume:
- After finishing each λ, append results to a CSV (or write a temp file).
- If lambda_scores.csv exists, skip lambdas already computed.

lambda_scores.csv format (one row per lambda):
Columns:
- K (=3)
- model_type (= "CJM_mode")
- lambda
- T (=1000)
- mean_BAC
- std_BAC
- n_sim

best_lambda.json format:
{
  "K3": {
    "CJM_mode": <best_lambda_float>
  }
}

Plot:
- x-axis log λ, y-axis mean_BAC.
- Save to outputs/step3_lambda/bac_vs_lambda.png

========================
STEP 4 — FIT CJM ON REAL VNINDEX (K=3) USING SELECTED λ (src/step4_fit_cjm_realdata_K3.py)
========================
Fit on real features (NOT returns-only):
- X_real is (T, F) standardized indicator matrix from Step 0.
- ret_ser is daily log return vector (for sorting/diagnostics).

Use JumpModel:
cjm_real = JumpModel(
  n_components=3,
  jump_penalty=lambda_star,
  cont=True,
  mode_loss=True,
  grid_size=0.05,
  n_init=5,              # can be slightly higher for real data
  max_iter=1000,
  tol=1e-8,
  random_state=123
)
Fit:
- cjm_real.fit(X_real, ret_ser=ret, sort_by="cumret")  # ensure stable label ordering

Outputs:
Create outputs/step4_apply/regimes_daily.csv with:
- date
- ret
- label = argmax(proba) (0..2)
- proba_0, proba_1, proba_2
Optional extra columns (nice-to-have but not required):
- center_{k}_{feature} for k=0..2 OR a separate JSON for centers
- inferred transition matrix (3×3) as a separate JSON

Also produce a short printed summary:
- regime counts and average duration
- mean/vol of returns per regime
- transition matrix

========================
QUALITY / VALIDATION CHECKLIST (must do)
========================
1) Reproducibility: fixed seeds; deterministic outputs.
2) Correct shapes: jumpmodels and hmmlearn require 2D X.
3) BAC correctness: must be permutation-invariant for K=3.
4) Sanity plot: bac_vs_lambda.png should be smooth-ish vs log λ.
5) Runtime: Ensure FAST MODE runs end-to-end on a Mac Air M4 without exhausting memory.

========================
DELIVERABLES (definition of done)
========================
- outputs/step1_params/hmm_K3.json
- outputs/step2_sim/K3/ (simulated datasets)
- outputs/step3_lambda/lambda_scores.csv
- outputs/step3_lambda/best_lambda.json
- outputs/step3_lambda/bac_vs_lambda.png
- outputs/step4_apply/regimes_daily.csv

When you write code:
- Keep scripts modular and runnable.
- Use clear logging and progress bars.
- Add minimal README instructions (how to run step-by-step).
- Do not compute indicators; assume they exist.
- Do not ask the user questions; make reasonable defaults but keep them in config YAML.
