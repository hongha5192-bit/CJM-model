#!/usr/bin/env python
"""
Fast Mode Runner for K=3 CJM Pipeline
Optimized for Mac Air M4 with reduced computational load
"""
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

# Import all steps
from step0_prepare_vnindex import prepare_vnindex_data
from step1_fit_hmm_K3 import fit_hmm_K3
from step2_simulate_hmm_K3 import simulate_hmm_K3
from step3_lambda_scan_K3 import lambda_scan_K3
from step4_fit_cjm_realdata_K3 import fit_cjm_realdata_K3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_step(step_name, step_func, config_or_path):
    """Run a single step with timing and error handling"""
    print("\n" + "="*80)
    print(f"Running {step_name}...")
    print("="*80)

    start_time = time.time()

    try:
        result = step_func(config_or_path)
        elapsed = time.time() - start_time
        print(f"✓ {step_name} completed in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error in {step_name}: {e}")
        logger.error(f"{step_name} failed after {elapsed:.2f} seconds", exc_info=True)
        raise


def main():
    """Run complete K=3 CJM pipeline with Fast Mode optimizations"""

    print("="*80)
    print("CONTINUOUS JUMP MODEL (CJM) K=3 - FAST MODE")
    print("Optimized for Mac Air M4")
    print("="*80)

    config_path = 'configs/daily_K3.yaml'

    # Check if config exists
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found at {config_path}")
        return 1

    # Load config for step 0
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    total_start = time.time()

    try:
        # Step 0: Prepare data (uses existing prepared data if available)
        data_path = Path('outputs/prepared_data.csv')
        if data_path.exists():
            print("\n✓ Step 0: Using existing prepared data")
        else:
            run_step("Step 0: Data Preparation", prepare_vnindex_data, config)

        # Step 1: Fit HMM with K=3
        hmm_path = Path('outputs/step1_params/hmm_K3.json')
        if hmm_path.exists():
            print("\n✓ Step 1: Using existing HMM K=3 parameters")
            print(f"  Delete {hmm_path} to refit")
        else:
            run_step("Step 1: Fit HMM K=3", fit_hmm_K3, config_path)

        # Step 2: Generate simulations (Fast mode: N=256)
        sim_dir = Path('outputs/step2_sim/K3')
        if sim_dir.exists() and len(list(sim_dir.glob('sim_T*.parquet'))) > 0:
            n_existing = len(list(sim_dir.glob('sim_T*.parquet')))
            print(f"\n✓ Step 2: Found {n_existing} existing simulations")
            print(f"  Delete {sim_dir}/* to regenerate")
        else:
            run_step("Step 2: Generate Simulations (N=256)", simulate_hmm_K3, config_path)

        # Step 3: Lambda scan (can resume from partial results)
        lambda_path = Path('outputs/step3_lambda/best_lambda_K3.json')
        if lambda_path.exists():
            print("\n✓ Step 3: Using existing lambda selection")
            print(f"  Delete {lambda_path} to rescan")
        else:
            run_step("Step 3: Lambda Scan (21 points)", lambda_scan_K3, config_path)

        # Step 4: Apply CJM to real data
        regimes_path = Path('outputs/step4_apply/regimes_daily_K3.csv')
        if regimes_path.exists():
            print("\n✓ Step 4: Regime predictions already exist")
            print(f"  Delete {regimes_path} to refit")
        else:
            run_step("Step 4: Fit CJM K=3 on Real Data", fit_cjm_realdata_K3, config_path)

        # Summary
        total_elapsed = time.time() - total_start
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Total runtime: {total_elapsed:.2f} seconds ({total_elapsed/60:.1f} minutes)")
        print("\nOutput files:")
        print("  - outputs/step1_params/hmm_K3.json")
        print("  - outputs/step2_sim/K3/ (256 simulations)")
        print("  - outputs/step3_lambda/lambda_scores_K3.csv")
        print("  - outputs/step3_lambda/best_lambda_K3.json")
        print("  - outputs/step3_lambda/bac_vs_lambda_K3.png")
        print("  - outputs/step4_apply/regimes_daily_K3.csv")
        print("  - outputs/step4_apply/regime_analysis_K3.png")

        return 0

    except Exception as e:
        total_elapsed = time.time() - total_start
        print("\n" + "="*80)
        print("PIPELINE FAILED")
        print("="*80)
        print(f"Error after {total_elapsed:.2f} seconds: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())