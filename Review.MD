2) Step-by-step review: what’s correct + what needs fixing
Step 1 — step1_fit_hmm_K3.py
What’s good
Reads outputs/prepared_data.csv, uses ret column, fits a Gaussian HMM, writes outputs/step1_params/hmm_K3.json.
Has the same output checks that your MD requires: P rows sum to 1, sigma>0, pi sums to 1. 
jump_model_lambda_simulation_vi…
Issues / fixes
init_params='kmeans++' is almost certainly invalid for hmmlearn
Most hmmlearn APIs expect init_params like 'stmc' (not “kmeans++”). If your code runs at all, it’s either ignoring it or you’re using a modified fork. If it errors, this is why Step 1 fails.
“Stationary distribution” is not computed correctly
pi = model.startprob_.tolist()
That’s the initial distribution, not necessarily the stationary distribution.
Your MD simulation step explicitly says draw s_1 ~ Categorical(pi) where pi is the stationary distribution. 
jump_model_lambda_simulation_vi…
✅ Fix: compute π from P via solving π = πP (left eigenvector of P with eigenvalue 1), then normalize.
Log-likelihood is wrong
'log_likelihood': float(model.score(X) * len(X)),
Usually .score(X) already returns total log-likelihood for the sequence, not per-observation, so multiplying by len(X) inflates it.
Sorting is inconsistent
You comment “sorted by volatility” but sort by abs(mean):
sorted_idx = np.argsort([abs(m) for m in means])
Not fatal (because you also do permutation-fix later in scoring), but confusing.
----
Step 2 — step2_simulate_hmm_K3.py
What’s good
Simulation procedure matches the MD: draw initial state from pi, transition via P, emit y_t ~ Normal(mu[state], sigma[state]^2); save (t, y, s_true). 
jump_model_lambda_simulation_vi…
Output format: Parquet per sim file, good.
Minor issue
n_switches = (df['s_true'].diff() != 0).sum() counts the first row as a “switch” because NaN != 0 is True. Cosmetic only.
-----
Step 3 — step3_lambda_scan_K3.py
What’s good
Implements BAC with permutation fix for K=3 (all 6 permutations) exactly as required. 
jump_model_lambda_simulation_vi…
Scans log-spaced λ grid (size defined in config), selects best λ by mean BAC, saves CSV and JSON.
Differences vs MD
Your MD suggests logging model_type in {JM, CJM_mode} and optionally also CJM_no_mode and saving per (K, model_type, lambda). 
jump_model_lambda_simulation_vi…

Your code currently scans only CJM_mode (cont=True, mode_loss=True). That’s fine if you intentionally only want CJM_mode.
Potential efficiency issues
It reloads every sim file from disk for every λ; not the main bottleneck, but it adds overhead.
----
