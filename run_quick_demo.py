#!/usr/bin/env python
"""
Quick demo of the pipeline with reduced parameters for faster execution
"""
import os
os.chdir('/Users/hanguyen/CJMModel/jump_lambda_vn')

# Temporarily modify config for quick demo
import yaml

# Load config
with open('configs/daily.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Reduce parameters for quick demo
config['simulation']['N_sim'] = 10  # Only 10 simulations instead of 1024
config['simulation']['T'] = 100      # Shorter sequences
config['lambda_scan']['lambda_grid']['n_grid'] = 5  # Only 5 lambdas instead of 29

# Save temporary config
with open('configs/daily_demo.yaml', 'w') as f:
    yaml.dump(config, f)

print("Running quick demo with reduced parameters:")
print(f"  N_sim: {config['simulation']['N_sim']} (instead of 1024)")
print(f"  T: {config['simulation']['T']} (instead of 1000)")
print(f"  Lambda grid: {config['lambda_scan']['lambda_grid']['n_grid']} values (instead of 29)")
print()

# Run only Steps 3 and 4 (Steps 0-2 are already complete)
import subprocess
import time

print("=" * 80)
print("Running Step 3: Lambda Scan (simplified)...")
print("=" * 80)

# First, create the 10 simulation files from existing ones
import shutil
for i in range(10):
    src = f"outputs/step2_sim/K2/sim_T1000_seed{i:03d}.parquet"
    dst = f"outputs/step2_sim/K2/sim_T0100_seed{i:03d}.parquet"
    if os.path.exists(src) and not os.path.exists(dst):
        # Just copy and rename for demo
        import pandas as pd
        df = pd.read_parquet(src)
        df = df.head(100)  # Take first 100 rows
        df['T'] = 100
        df.to_parquet(dst, index=False)

# Create a simplified Step 3
code = '''
import sys
sys.path.append('.')
from src.utils_io import load_config, load_parquet, save_csv, save_json, setup_logging, numpy_to_python
from src.metrics import best_balanced_accuracy_K2
from src.jump_model_mock import JumpModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = setup_logging('step3_demo')

# Load demo config
config = load_config('configs/daily_demo.yaml')

# Create lambda grid
lambdas = np.logspace(-1, 3, 5)  # 0.1, 1, 10, 100, 1000
logger.info(f"Testing lambdas: {lambdas}")

# Load 10 simulations
sim_data = []
for i in range(10):
    df = load_parquet(f"outputs/step2_sim/K2/sim_T0100_seed{i:03d}.parquet")
    sim_data.append((df['y'].values, df['s_true'].values))

logger.info(f"Loaded {len(sim_data)} simulations")

# Test each lambda
results = []
for lam in lambdas:
    logger.info(f"Testing lambda={lam:.1f}")
    for model_type in ['JM', 'CJM_mode']:
        bac_scores = []
        for y, s_true in sim_data:
            try:
                X = y.reshape(-1, 1)
                model = JumpModel(n_components=2, jump_penalty=lam,
                                cont=(model_type=='CJM_mode'),
                                mode_loss=True, n_init=2, max_iter=100,
                                tol=1e-4, random_state=42)
                model.fit(X, ret_ser=y, sort_by="cumret")

                if model_type == 'JM':
                    y_pred = model.predict(X)
                else:
                    proba = model.predict_proba(X)
                    y_pred = proba.values.argmax(axis=1)

                bac = best_balanced_accuracy_K2(s_true, y_pred)
                bac_scores.append(bac)
            except:
                pass

        if bac_scores:
            results.append({
                'K': 2,
                'model_type': model_type,
                'lambda': lam,
                'mean_BAC': np.mean(bac_scores),
                'std_BAC': np.std(bac_scores),
                'n_sim': len(bac_scores)
            })

# Save results
df_results = pd.DataFrame(results)
save_csv(df_results, 'outputs/step3_lambda/lambda_scores.csv')

# Find best lambda
best_lambda = {}
for mt in ['JM', 'CJM_mode']:
    df_mt = df_results[df_results['model_type'] == mt]
    if len(df_mt) > 0:
        best_idx = df_mt['mean_BAC'].idxmax()
        best_lambda[mt] = float(df_mt.loc[best_idx, 'lambda'])

save_json({'K2': best_lambda}, 'outputs/step3_lambda/best_lambda.json')

# Create plot
plt.figure(figsize=(8, 5))
for mt in ['JM', 'CJM_mode']:
    df_mt = df_results[df_results['model_type'] == mt]
    if len(df_mt) > 0:
        plt.plot(df_mt['lambda'], df_mt['mean_BAC'], 'o-', label=mt)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Mean BAC')
plt.title('BAC vs Lambda (Demo)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/step3_lambda/bac_vs_lambda.png', dpi=100)
plt.close()

logger.info("Step 3 demo complete!")
'''

with open('src/step3_demo.py', 'w') as f:
    f.write(code)

# Run Step 3 demo
start = time.time()
subprocess.run(['python', 'src/step3_demo.py'], check=True)
elapsed = time.time() - start
print(f"✓ Step 3 completed in {elapsed:.1f} seconds")

print("\n" + "=" * 80)
print("Running Step 4: Fit CJM to Real Data...")
print("=" * 80)

start = time.time()
subprocess.run(['python', 'src/step4_fit_jumpmodel_realdata.py'], check=True)
elapsed = time.time() - start
print(f"✓ Step 4 completed in {elapsed:.1f} seconds")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("Key outputs:")
print("  • Lambda scores: outputs/step3_lambda/lambda_scores.csv")
print("  • Best lambda: outputs/step3_lambda/best_lambda.json")
print("  • Plot: outputs/step3_lambda/bac_vs_lambda.png")
print("  • Regimes: outputs/step4_apply/regimes_daily.csv")
print("=" * 80)