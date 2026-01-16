#!/usr/bin/env python
"""
Main runner script to execute all steps of the CJM project
"""
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append('src')


def run_step(step_name, script_path):
    """Run a single step"""
    print("=" * 80)
    print(f"Running {step_name}...")
    print("=" * 80)

    start_time = time.time()

    try:
        # Run the script directly using exec
        with open(script_path, 'r') as f:
            code = f.read()

        # Create a namespace for execution
        namespace = {'__name__': '__main__'}
        exec(code, namespace)

        elapsed = time.time() - start_time
        print(f"✓ {step_name} completed in {elapsed:.2f} seconds")

    except Exception as e:
        print(f"✗ Error in {step_name}: {e}")
        raise

    print()


def main():
    """Execute all steps in sequence"""
    print("\n" + "=" * 80)
    print("CONTINUOUS JUMP MODEL (CJM) FOR VNINDEX")
    print("Running complete pipeline...")
    print("=" * 80 + "\n")

    total_start = time.time()

    # Define steps
    steps = [
        ("Step 0: Data Preparation", "src/step0_prepare_vnindex.py"),
        ("Step 1: Fit HMM", "src/step1_fit_hmm.py"),
        ("Step 2: Simulate Sequences", "src/step2_simulate_hmm.py"),
        ("Step 3: Lambda Scan", "src/step3_lambda_scan.py"),
        ("Step 4: Fit CJM to Real Data", "src/step4_fit_jumpmodel_realdata.py"),
    ]

    # Run each step
    for step_name, script_name in steps:
        run_step(step_name, script_name)

    # Summary
    total_elapsed = time.time() - total_start
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total execution time: {total_elapsed:.2f} seconds")
    print("=" * 80)

    # Print output locations
    print("\nKey outputs generated:")
    print("  • HMM parameters: outputs/step1_params/hmm_K2.json")
    print("  • Simulations: outputs/step2_sim/K2/")
    print("  • Lambda scores: outputs/step3_lambda/lambda_scores.csv")
    print("  • Best lambda: outputs/step3_lambda/best_lambda.json")
    print("  • BAC plot: outputs/step3_lambda/bac_vs_lambda.png")
    print("  • Final regimes: outputs/step4_apply/regimes_daily.csv")
    print()


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not Path('configs/daily.yaml').exists():
        print("Error: Please run this script from the jump_lambda_vn directory")
        sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)