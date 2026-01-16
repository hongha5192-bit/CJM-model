#!/usr/bin/env python
"""
Step 3: Optimized Lambda scan for CJM K=3 - Mac Air M4
Implements all optimizations from Improvements.MD:
- Loads simulations into memory once (avoid repeated disk reads)
- Uses optimized parameters (N_sim=128, n_grid=11, n_jobs=2)
- grid_size=0.05 for K=3 as per paper
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from jumpmodels.jump import JumpModel
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
import warnings
import gc
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


def fit_cjm_single_sim_memory(sim_data, lam, K, grid_size, n_init, max_iter, tol):
    """Fit CJM to simulation data already in memory"""
    try:
        y = sim_data['y']
        s_true = sim_data['s_true']

        # Prepare data for JumpModel (must be 2D)
        X = y.reshape(-1, 1)

        # Fit CJM with mode loss
        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,
            mode_loss=True,
            grid_size=grid_size,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
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
        logger.warning(f"Failed to fit simulation: {e}")
        return np.nan


def load_all_simulations(sim_dir, max_sims=128):
    """Load all simulations into memory once"""
    logger.info(f"Loading simulations into memory (max {max_sims})...")

    sim_files = sorted(sim_dir.glob('sim_T*.parquet'))[:max_sims]

    if not sim_files:
        sim_files = sorted(sim_dir.glob('sim_T*.csv'))[:max_sims]

    simulations = []

    for sim_path in tqdm(sim_files, desc="Loading simulations"):
        if sim_path.suffix == '.parquet':
            df = pd.read_parquet(sim_path)
        else:
            df = pd.read_csv(sim_path)

        simulations.append({
            'y': df['y'].values,
            's_true': df['s_true'].values,
            'sim_id': df['sim_id'].iloc[0]
        })

    logger.info(f"Loaded {len(simulations)} simulations into memory")
    return simulations


def lambda_scan_K3_optimized(config_path='configs/daily_K3.yaml'):
    """Optimized lambda scan for Mac Air M4"""

    # Load config
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = config['hmm']['K']
    assert K == 3, f"Expected K=3, got K={K}"

    # Mac Air M4 optimized parameters
    N_sim = config['simulation']['N_sim']  # Should be 128
    log10_min = config['lambda_scan']['lambda_grid']['log10_min']
    log10_max = config['lambda_scan']['lambda_grid']['log10_max']
    n_grid = config['lambda_scan']['lambda_grid']['n_grid']  # Should be 11

    lambda_grid = np.logspace(log10_min, log10_max, n_grid)

    logger.info("=" * 60)
    logger.info("MAC AIR M4 OPTIMIZED LAMBDA SCAN")
    logger.info("=" * 60)
    logger.info(f"K = {K}")
    logger.info(f"N_sim = {N_sim} (optimized)")
    logger.info(f"Lambda grid: {n_grid} points from {lambda_grid[0]:.2e} to {lambda_grid[-1]:.2e}")

    # CJM settings (Mac Air M4 optimized)
    grid_size = config['lambda_scan']['cjm']['grid_size']  # 0.05
    n_init = config['lambda_scan']['jumpmodels']['n_init']  # 3
    max_iter = config['lambda_scan']['jumpmodels']['max_iter']  # 200
    tol = config['lambda_scan']['jumpmodels']['tol']  # 1e-5
    n_jobs = config['lambda_scan']['jumpmodels'].get('n_jobs', 2)  # 2

    logger.info(f"\nOptimized CJM settings:")
    logger.info(f"  grid_size = {grid_size} (K=3 paper spec)")
    logger.info(f"  n_init = {n_init}")
    logger.info(f"  max_iter = {max_iter}")
    logger.info(f"  tol = {tol}")
    logger.info(f"  n_jobs = {n_jobs} (optimized for Mac Air M4)")

    # Load all simulations into memory once (Improvement D)
    sim_dir = Path(f'outputs/step2_sim/K{K}')
    simulations = load_all_simulations(sim_dir, max_sims=N_sim)

    if len(simulations) == 0:
        raise ValueError("No simulations loaded")

    # Create output directory
    output_path = Path('outputs/step3_lambda')
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    scores_path = output_path / 'lambda_scores_K3_optimized.csv'

    # Main lambda scan loop
    logger.info("\nStarting optimized lambda scan...")
    logger.info("=" * 60)

    for lam_idx, lam in enumerate(lambda_grid):
        logger.info(f"\nProcessing λ={lam:.2e} ({lam_idx+1}/{n_grid})...")

        # Fit CJM to all simulations in parallel (using data from memory)
        bac_scores = Parallel(n_jobs=n_jobs)(
            delayed(fit_cjm_single_sim_memory)(
                sim_data, lam, K, grid_size, n_init, max_iter, tol
            )
            for sim_data in tqdm(simulations, desc=f"λ={lam:.2e}", leave=False)
        )

        # Filter out failed fits
        bac_scores = [s for s in bac_scores if not np.isnan(s)]

        if len(bac_scores) == 0:
            logger.warning(f"All fits failed for λ={lam:.2e}")
            continue

        # Calculate statistics
        mean_bac = np.mean(bac_scores)
        std_bac = np.std(bac_scores)

        logger.info(f"  Result: BAC = {mean_bac:.4f} ± {std_bac:.4f} (n={len(bac_scores)}/{len(simulations)})")

        # Store result
        result = {
            'K': K,
            'model_type': 'CJM_mode',
            'lambda': lam,
            'T': 1000,
            'mean_BAC': mean_bac,
            'std_BAC': std_bac,
            'n_success': len(bac_scores),
            'n_sim': len(simulations),
            'optimization': 'Mac_Air_M4'
        }
        results.append(result)

        # Save incrementally
        df_results = pd.DataFrame(results)
        df_results.to_csv(scores_path, index=False)

        # Force garbage collection to free memory
        gc.collect()

    # Final results analysis
    logger.info("\n" + "=" * 60)
    logger.info("LAMBDA SCAN COMPLETE")
    logger.info("=" * 60)

    df_final = pd.DataFrame(results)

    # Find best lambda
    best_idx = df_final['mean_BAC'].idxmax()
    best_row = df_final.loc[best_idx]
    best_lambda = best_row['lambda']
    best_bac = best_row['mean_BAC']

    logger.info(f"\nBest lambda: {best_lambda:.2e}")
    logger.info(f"Best BAC: {best_bac:.4f} ± {best_row['std_BAC']:.4f}")

    # Save best lambda
    best_lambda_dict = {
        f'K{K}': {
            'CJM_mode': float(best_lambda),
            'mean_BAC': float(best_bac),
            'std_BAC': float(best_row['std_BAC']),
            'n_simulations': len(simulations),
            'optimization': 'Mac_Air_M4',
            'parameters': {
                'grid_size': grid_size,
                'n_init': n_init,
                'max_iter': max_iter,
                'tol': tol,
                'n_jobs': n_jobs
            }
        }
    }

    best_lambda_path = output_path / 'best_lambda_K3_optimized.json'
    with open(best_lambda_path, 'w') as f:
        json.dump(best_lambda_dict, f, indent=2)

    logger.info(f"Saved best lambda to {best_lambda_path}")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot BAC vs lambda
    ax.errorbar(
        df_final['lambda'],
        df_final['mean_BAC'],
        yerr=df_final['std_BAC'],
        marker='o',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        label=f'CJM K=3 (n={len(simulations)})'
    )

    # Mark best lambda
    ax.axvline(x=best_lambda, color='red', linestyle='--', alpha=0.7,
               label=f'Best λ={best_lambda:.2e}')

    # Add horizontal line at random performance
    ax.axhline(y=1/K, color='gray', linestyle=':', alpha=0.5,
               label=f'Random (BAC={1/K:.3f})')

    ax.set_xscale('log')
    ax.set_xlabel('Lambda (λ)', fontsize=12)
    ax.set_ylabel('Mean Balanced Accuracy', fontsize=12)
    ax.set_title(f'Lambda Selection K={K} (Mac Air M4 Optimized)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Add text box with optimization details
    textstr = f'Best λ = {best_lambda:.2e}\nBAC = {best_bac:.4f} ± {best_row["std_BAC"]:.4f}\n'
    textstr += f'N_sim = {len(simulations)}, n_grid = {n_grid}\n'
    textstr += f'grid_size = {grid_size}, n_jobs = {n_jobs}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plot_path = output_path / 'bac_vs_lambda_K3_optimized.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Saved plot to {plot_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("LAMBDA SCAN RESULTS (MAC AIR M4 OPTIMIZED)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  K = {K} states")
    print(f"  N_sim = {len(simulations)} simulations")
    print(f"  Lambda points = {n_grid}")
    print(f"  grid_size = {grid_size} (K=3 paper spec)")
    print(f"  n_jobs = {n_jobs} (optimized for Mac Air M4)")
    print(f"\nResults Table:")
    print("-" * 50)
    print(f"{'Lambda':>10} | {'Mean BAC':>10} | {'Std BAC':>10}")
    print("-" * 50)
    for _, row in df_final.iterrows():
        print(f"{row['lambda']:10.2e} | {row['mean_BAC']:10.4f} | {row['std_BAC']:10.4f}")
    print("-" * 50)
    print(f"\n★ BEST: λ = {best_lambda:.2e} → BAC = {best_bac:.4f}")
    print(f"  Improvement over random: {(best_bac - 1/K)/(1/K)*100:.1f}%")
    print("=" * 70)

    return best_lambda


if __name__ == "__main__":
    try:
        lambda_scan_K3_optimized()
    except Exception as e:
        logger.error(f"Error in optimized lambda scan: {e}", exc_info=True)
        raise