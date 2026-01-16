#!/usr/bin/env python
"""
Step 3: Lambda scan for CJM K=3 - Ultra fast version
Uses only 10 simulations and 7 lambda points for quick results
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from jumpmodels.jump import JumpModel
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def best_balanced_accuracy_K3(y_true, y_pred):
    """
    Compute best balanced accuracy for K=3 with permutation fix.
    Tries all 6 permutations of {0, 1, 2}, returns the max BAC.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0

    # Try all 6 permutations of {0, 1, 2}
    for perm in permutations([0, 1, 2]):
        # Create mapping
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}

        # Apply mapping to predictions
        y_pred_mapped = np.array([mapping[label] for label in y_pred])

        # Calculate BAC for this permutation
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def fit_cjm_single_sim(sim_path, lam, K, grid_size):
    """Fit CJM to a single simulation and return BAC - Simplified version"""
    try:
        # Load simulation
        if sim_path.suffix == '.parquet':
            df_sim = pd.read_parquet(sim_path)
        else:
            df_sim = pd.read_csv(sim_path)

        y = df_sim['y'].values
        s_true = df_sim['s_true'].values

        # Prepare data for JumpModel (must be 2D)
        X = y.reshape(-1, 1)

        # Fit CJM with reduced settings for speed
        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,
            mode_loss=True,
            grid_size=grid_size,
            n_init=1,      # Reduced from 3
            max_iter=100,  # Reduced from 500
            tol=1e-4,      # Relaxed from 1e-6
            random_state=123,
            verbose=0
        )

        cjm.fit(X)

        # Get predictions
        s_pred = cjm.predict(X)

        # Compute permutation-invariant BAC
        bac = best_balanced_accuracy_K3(s_true, s_pred)

        return bac

    except Exception as e:
        logger.warning(f"Failed to fit {sim_path.name}: {e}")
        return np.nan


def lambda_scan_K3_ultra_fast():
    """Ultra fast lambda scan for K=3 - demonstration purposes"""

    logger.info("Starting ULTRA FAST lambda scan for K=3")
    logger.info("Using minimal settings for quick demonstration")

    K = 3
    grid_size = 0.05

    # Use only 7 lambda points for ultra fast scan
    lambda_grid = np.logspace(-1, 3, 7)  # 0.1 to 1000, 7 points

    logger.info(f"Lambda grid (ultra fast): {len(lambda_grid)} points")
    logger.info(f"Lambda values: {', '.join([f'{l:.2e}' for l in lambda_grid])}")

    # Get simulation files - USE ONLY 10
    sim_dir = Path(f'outputs/step2_sim/K{K}')
    sim_files = sorted(sim_dir.glob('sim_T*.parquet'))[:10]

    N_sim = len(sim_files)
    logger.info(f"Using only {N_sim} simulations for ultra-fast results")

    # Create output directory
    output_path = Path('outputs/step3_lambda')
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # Sequential processing for simplicity
    logger.info("Starting lambda scan (sequential processing)...")

    for lam_idx, lam in enumerate(lambda_grid):
        logger.info(f"\nProcessing λ={lam:.2e} ({lam_idx+1}/{len(lambda_grid)})...")

        bac_scores = []
        for sim_idx, sim_path in enumerate(sim_files):
            print(f"  Simulation {sim_idx+1}/{N_sim}...", end='')

            bac = fit_cjm_single_sim(sim_path, lam, K, grid_size)

            if not np.isnan(bac):
                bac_scores.append(bac)
                print(f" BAC={bac:.3f}")
            else:
                print(" Failed")

        if len(bac_scores) > 0:
            mean_bac = np.mean(bac_scores)
            std_bac = np.std(bac_scores)
            logger.info(f"  λ={lam:.2e}: Mean BAC = {mean_bac:.4f} ± {std_bac:.4f}")

            result = {
                'lambda': lam,
                'mean_BAC': mean_bac,
                'std_BAC': std_bac,
                'n_success': len(bac_scores),
                'n_sim': N_sim
            }
            results.append(result)

    if len(results) == 0:
        logger.error("No successful fits!")
        return None

    # Create results dataframe
    df_results = pd.DataFrame(results)

    # Find best lambda
    best_idx = df_results['mean_BAC'].idxmax()
    best_row = df_results.loc[best_idx]
    best_lambda = best_row['lambda']
    best_bac = best_row['mean_BAC']

    # Save results
    scores_path = output_path / 'lambda_scores_K3_ultra_fast.csv'
    df_results.to_csv(scores_path, index=False)

    # Save best lambda
    best_lambda_dict = {
        'K3': {
            'best_lambda': float(best_lambda),
            'mean_BAC': float(best_bac),
            'std_BAC': float(best_row['std_BAC']),
            'n_simulations': N_sim,
            'note': 'Ultra fast scan - demonstration only'
        }
    }

    best_lambda_path = output_path / 'best_lambda_K3_ultra_fast.json'
    with open(best_lambda_path, 'w') as f:
        json.dump(best_lambda_dict, f, indent=2)

    # Create simple plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        df_results['lambda'],
        df_results['mean_BAC'],
        yerr=df_results['std_BAC'],
        marker='o',
        markersize=10,
        linewidth=2,
        capsize=5,
        label=f'K=3 Ultra Fast ({N_sim} sims)'
    )

    ax.axvline(x=best_lambda, color='red', linestyle='--',
               label=f'Best λ={best_lambda:.2e}')
    ax.axhline(y=1/3, color='gray', linestyle=':',
               label='Random (0.333)')

    ax.set_xscale('log')
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Mean BAC', fontsize=12)
    ax.set_title(f'K=3 Lambda Selection (Ultra Fast: {N_sim} simulations)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = output_path / 'bac_vs_lambda_K3_ultra_fast.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("ULTRA FAST LAMBDA SCAN COMPLETE (K=3)")
    print("="*60)
    print(f"Simulations used: {N_sim}")
    print(f"Lambda points tested: {len(lambda_grid)}")
    print(f"\nResults:")
    print("-"*40)
    for _, row in df_results.iterrows():
        print(f"λ={row['lambda']:8.2e}: BAC={row['mean_BAC']:.4f} ± {row['std_BAC']:.4f}")
    print("-"*40)
    print(f"\nBest lambda: {best_lambda:.2e}")
    print(f"Best BAC: {best_bac:.4f} ± {best_row['std_BAC']:.4f}")
    print(f"Improvement over random: {(best_bac - 1/3)/(1/3)*100:.1f}%")
    print("="*60)
    print("\nFiles saved:")
    print(f"  - {scores_path}")
    print(f"  - {best_lambda_path}")
    print(f"  - {plot_path}")

    return best_lambda


if __name__ == "__main__":
    try:
        lambda_scan_K3_ultra_fast()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise