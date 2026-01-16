#!/usr/bin/env python
"""
Check pipeline status
"""
import os
import glob
import json
import time
import subprocess

print("=" * 60)
print("CJM PIPELINE STATUS")
print("=" * 60)

# Check if pipeline is running
result = subprocess.run(['pgrep', '-f', 'run_all.py'], capture_output=True, text=True)
if result.returncode == 0:
    print("🔄 Pipeline Status: RUNNING (PID: {})".format(result.stdout.strip()))
else:
    print("⏹️  Pipeline Status: NOT RUNNING")

print("\nStep Progress:")
print("-" * 40)

# Step 0: Data prep
if os.path.exists('outputs/prepared_data.csv'):
    print("✅ Step 0: Data Preparation - COMPLETE")
else:
    print("⏳ Step 0: Data Preparation - PENDING")

# Step 1: HMM
if os.path.exists('outputs/step1_params/hmm_K2.json'):
    with open('outputs/step1_params/hmm_K2.json', 'r') as f:
        hmm = json.load(f)
    print("✅ Step 1: HMM Fitting - COMPLETE")
    print(f"   States: {hmm['K']}, Converged: {hmm['convergence']['converged']}")
else:
    print("⏳ Step 1: HMM Fitting - PENDING")

# Step 2: Simulations
sim_files = glob.glob('outputs/step2_sim/K2/*.parquet')
n_sims = len(sim_files)
if n_sims > 0:
    if n_sims >= 1024:
        print(f"✅ Step 2: Simulations - COMPLETE ({n_sims} files)")
    else:
        print(f"🔄 Step 2: Simulations - IN PROGRESS ({n_sims}/1024)")
else:
    print("⏳ Step 2: Simulations - PENDING")

# Step 3: Lambda scan
if os.path.exists('outputs/step3_lambda/lambda_scores.csv'):
    print("✅ Step 3: Lambda Scan - COMPLETE")
    if os.path.exists('outputs/step3_lambda/best_lambda.json'):
        with open('outputs/step3_lambda/best_lambda.json', 'r') as f:
            best = json.load(f)
        if 'K2' in best:
            print(f"   Best λ (JM): {best['K2'].get('JM', 'N/A')}")
            print(f"   Best λ (CJM): {best['K2'].get('CJM_mode', 'N/A')}")
else:
    print("🔄 Step 3: Lambda Scan - IN PROGRESS")
    print("   This step tests 29 λ values × 1024 sims × 2 models")
    print("   Expected time: 30-60 minutes with real JumpModel")

# Step 4: Real data
if os.path.exists('outputs/step4_apply/regimes_daily.csv'):
    import pandas as pd
    df = pd.read_csv('outputs/step4_apply/regimes_daily.csv')
    print(f"✅ Step 4: CJM on Real Data - COMPLETE")
    print(f"   Days classified: {len(df)}")
    if 'label' in df.columns:
        counts = df['label'].value_counts()
        print(f"   Regime 0: {counts.get(0, 0)} days")
        print(f"   Regime 1: {counts.get(1, 0)} days")
else:
    print("⏳ Step 4: CJM on Real Data - PENDING")

print("\n" + "=" * 60)

# Estimate time remaining
if n_sims >= 1024 and not os.path.exists('outputs/step3_lambda/lambda_scores.csv'):
    print("⏱️  Estimated time remaining: 20-40 minutes")
    print("   (Lambda scan with real JumpModel is compute-intensive)")
    print("\n💡 Tip: Check full_pipeline.log for detailed progress")

print("=" * 60)