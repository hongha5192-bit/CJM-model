4) What to change (Mac Air M4 feasibility recommendations)
If you want this to be realistic on Mac Air M4, the biggest wins are:
A) Use the paper’s K=3 grid_size
Set in configs/daily_K3.yaml:
lambda_scan.cjm.grid_size: 0.05 ✅ (paper’s three-state choice) 
ssrn-4556048
keep mode_loss: True
This is the most important fix.
---
B) Reduce compute in lambda scan (without breaking the study)
Since you only need 1 simulation study, you can be aggressive:
N_sim: 64 or 128 (instead of 256)
n_grid: 11 (instead of 21 or 29)
n_init: 3–5 during scan (keep 10 only for the final real-data fit if you want robustness)
max_iter: 200–300, tol: 1e-5 (instead of 1e-8)
Your MD recommends a 29-point grid to replicate Figure-like behavior, but you explicitly said you only need one simulation study and have compute limits. 
----
C) Limit parallelism (counterintuitive but important on laptops)
Running n_jobs=4 processes with big DP matrices can cause:
memory pressure
CPU throttling
slower overall runtime
On Mac Air, often:
n_jobs=2 is faster than 4 for heavy allocations.
----
D) Avoid repeated disk reads in Step 3 (secondary but helpful)
Load simulations once into memory, then fit CJM using arrays rather than reading parquet/csv 5000+ times.
This won’t solve the DP cost problem, but it reduces overhead.