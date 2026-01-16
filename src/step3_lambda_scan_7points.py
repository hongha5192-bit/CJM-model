#!/usr/bin/env python
"""
Step 3: Lambda scan with 7 points using corrected simulations
Tests lambdas from 1e-2 to 1e5 using 512 corrected simulations
Optimized for Mac Air M4 - FAST VERSION
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def best_balanced_accuracy_K3(y_true, y_pred):
    """Compute best BAC for K=3 with permutation fix"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    best_bac = 0.0
    for perm in permutations([0, 1, 2]):
        mapping = {0: perm[0], 1: perm[1], 2: perm[2]}
        y_pred_mapped = np.array([mapping[label] for label in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def fit_cjm_single(sim_path, lam, K, feature_names, grid_size=0.05):
    """Fit CJM to a single simulation and compute BAC"""
    try:
        # Load simulation
        df = pd.read_parquet(sim_path)
        X = df[feature_names].values
        s_true = df['s_true'].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit CJM with mode loss - Mac Air M4 optimized settings
        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,              # Continuous model
            mode_loss=True,         # Mode loss version
            grid_size=grid_size,    # 0.05 for K=3
            n_init=2,               # Very reduced for Mac Air M4
            max_iter=300,           # Reduced for Mac Air M4
            tol=1e-5,              # Relaxed for Mac Air M4
            random_state=42,
            verbose=0
        )

        cjm.fit(X_scaled)
        s_pred = cjm.predict(X_scaled)

        # Compute BAC with permutation fix
        bac = best_balanced_accuracy_K3(s_true, s_pred)

        return bac

    except Exception as e:
        logger.debug(f"Failed to fit CJM for {sim_path}: {e}")
        return np.nan


def process_lambda_batch(lambda_batch, sim_files, K, feature_names, grid_size, n_jobs=2):
    """Process a batch of lambdas"""
    results = []

    for lam in lambda_batch:
        # Process simulations for this lambda
        bac_scores = Parallel(n_jobs=n_jobs)(
            delayed(fit_cjm_single)(sim_path, lam, K, feature_names, grid_size)
            for sim_path in sim_files
        )

        # Filter out NaN values
        valid_scores = [s for s in bac_scores if not np.isnan(s)]

        if len(valid_scores) > 0:
            mean_bac = np.mean(valid_scores)
            std_bac = np.std(valid_scores)
            n_valid = len(valid_scores)
        else:
            mean_bac = np.nan
            std_bac = np.nan
            n_valid = 0

        result = {
            'lambda': lam,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': n_valid,
            'n_total': len(sim_files)
        }

        results.append(result)

        # Log progress
        if not np.isnan(mean_bac):
            logger.info(f"λ={lam:10.2e}: BAC={mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(sim_files)} valid)")
        else:
            logger.info(f"λ={lam:10.2e}: BAC=NaN (all failed)")

    return results


def main():
    """Run lambda scan with 7 points optimized for Mac Air M4 using corrected simulations"""

    logger.info("="*80)
    logger.info("LAMBDA SCAN WITH 7 POINTS - FAST VERSION")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']
    grid_size = 0.05  # For K=3 as per paper

    # Lambda grid: 7 points from 1e-2 to 1e5 (more evenly spaced in log scale)
    lambda_grid = np.logspace(-2, 5, 7)

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Lambda grid: {len(lambda_grid)} points")
    logger.info(f"  Lambda range: {lambda_grid[0]:.2e} to {lambda_grid[-1]:.2e}")
    logger.info(f"  Grid size = {grid_size}")
    logger.info(f"  CJM settings: n_init=2, max_iter=300, tol=1e-5")

    # Load corrected simulations - use only first 100 for speed
    sim_dir = Path('outputs/step2_sim/K3_512_corrected')
    all_sim_files = sorted(sim_dir.glob('sim_*.parquet'))

    # Use subset for faster execution
    n_sims_to_use = 100  # Reduced for speed
    sim_files = all_sim_files[:n_sims_to_use]

    if not sim_files:
        logger.error(f"No simulation files found in {sim_dir}")
        return

    logger.info(f"Using {len(sim_files)} simulations (out of {len(all_sim_files)} available)")

    # Verify first simulation has correct ranges
    df_test = pd.read_parquet(sim_files[0])
    logger.info("\nVerifying simulation data ranges:")
    for feat in feature_names:
        logger.info(f"  {feat:10s}: mean={df_test[feat].mean():6.2f}, std={df_test[feat].std():6.2f}, "
                   f"min={df_test[feat].min():6.2f}, max={df_test[feat].max():6.2f}")

    # Number of parallel jobs (optimized for Mac Air M4)
    n_jobs = 2  # Conservative for Mac Air M4
    logger.info(f"\nUsing {n_jobs} parallel workers")

    # Calculate total fits
    total_fits = len(lambda_grid) * len(sim_files)
    logger.info(f"Total CJM fits to perform: {total_fits:,}")
    logger.info(f"Estimated time: {total_fits * 0.3 / 60:.1f} minutes (at ~0.3s per fit)")

    # Process lambdas in batches for better progress tracking
    batch_size = 2  # Process 2 lambdas at a time
    lambda_batches = [lambda_grid[i:i+batch_size] for i in range(0, len(lambda_grid), batch_size)]

    results = []
    start_time = time.time()

    logger.info("\nStarting lambda scan...")
    logger.info("-" * 60)

    for batch_idx, lambda_batch in enumerate(lambda_batches):
        batch_start = time.time()
        logger.info(f"\nProcessing batch {batch_idx+1}/{len(lambda_batches)} (λ values: {len(lambda_batch)})")

        batch_results = process_lambda_batch(
            lambda_batch, sim_files, K, feature_names, grid_size, n_jobs
        )
        results.extend(batch_results)

        batch_time = time.time() - batch_start
        logger.info(f"Batch {batch_idx+1} completed in {batch_time:.1f} seconds")

    elapsed = time.time() - start_time

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Find best lambda
    if not df_results['mean_bac'].isna().all():
        best_idx = df_results['mean_bac'].idxmax()
        best_lambda = df_results.loc[best_idx, 'lambda']
        best_bac = df_results.loc[best_idx, 'mean_bac']
        best_std = df_results.loc[best_idx, 'std_bac']
    else:
        logger.error("All lambda values resulted in NaN BAC!")
        best_lambda = lambda_grid[len(lambda_grid)//2]
        best_bac = np.nan
        best_std = np.nan

    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Average time per lambda: {elapsed/len(lambda_grid):.1f} seconds")
    logger.info(f"Average time per fit: {elapsed/total_fits:.3f} seconds")

    if not np.isnan(best_bac):
        logger.info(f"\n🎯 Best Lambda: {best_lambda:.2e}")
        logger.info(f"   Best Mean BAC: {best_bac:.4f} ± {best_std:.4f}")

    # Display all lambda results sorted by BAC
    logger.info("\nAll Lambda Values Sorted by BAC:")
    logger.info("-" * 70)
    logger.info(f"{'Rank':>4} | {'Lambda':>12} | {'Mean BAC':>10} | {'Std BAC':>10} | {'Valid':>7}")
    logger.info("-" * 70)

    df_sorted = df_results.sort_values('mean_bac', ascending=False).reset_index(drop=True)
    for idx, row in df_sorted.iterrows():
        if not np.isnan(row['mean_bac']):
            logger.info(f"{idx+1:4d} | {row['lambda']:12.2e} | {row['mean_bac']:10.4f} | "
                       f"{row['std_bac']:10.4f} | {row['n_valid']:4d}/{row['n_total']:<3d}")

    # Save results
    output_dir = Path('outputs/step3_lambda')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save lambda scores CSV
    csv_path = output_dir / 'lambda_scores_7points.csv'
    df_results['K'] = K
    df_results['model_type'] = 'CJM_mode'
    df_results['T'] = 1000
    df_results['N_sim'] = len(sim_files)
    df_results = df_results[['K', 'model_type', 'lambda', 'T', 'N_sim', 'mean_bac', 'std_bac', 'n_valid', 'n_total']]
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\n📊 Lambda scores saved to: {csv_path}")

    # Save best lambda JSON
    best_lambda_dict = {
        f"K{K}": {
            "CJM_mode": {
                "lambda": float(best_lambda),
                "mean_bac": float(best_bac) if not np.isnan(best_bac) else None,
                "std_bac": float(best_std) if not np.isnan(best_std) else None,
                "n_simulations": len(sim_files),
                "lambda_grid_size": len(lambda_grid),
                "simulation_dir": str(sim_dir)
            }
        }
    }
    json_path = output_dir / 'best_lambda_7points.json'
    with open(json_path, 'w') as f:
        json.dump(best_lambda_dict, f, indent=2)
    logger.info(f"📁 Best lambda saved to: {json_path}")

    # Create BAC vs Lambda plot
    create_bac_plot(df_results, best_lambda, best_bac, K, output_dir, len(sim_files))

    # Display final statistics
    logger.info("\n" + "="*80)
    logger.info("LAMBDA SCAN COMPLETE")
    logger.info("="*80)
    logger.info(f"Total simulations processed: {total_fits:,}")
    logger.info(f"Successful fits: {df_results['n_valid'].sum():,}")
    logger.info(f"Failed fits: {(df_results['n_total'].sum() - df_results['n_valid'].sum()):,}")
    logger.info(f"Success rate: {df_results['n_valid'].sum() / df_results['n_total'].sum() * 100:.1f}%")

    # Display lambda values tested
    logger.info("\nLambda values tested:")
    for i, lam in enumerate(lambda_grid):
        logger.info(f"  {i+1:2d}. λ = {lam:.2e}")

    return best_lambda, best_bac


def create_bac_plot(df_results, best_lambda, best_bac, K, output_dir, n_sims):
    """Create BAC vs Lambda plot"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Filter valid points
    valid_df = df_results[~df_results['mean_bac'].isna()].copy()

    if not valid_df.empty:
        # Main plot with error bars
        ax.errorbar(valid_df['lambda'], valid_df['mean_bac'],
                   yerr=valid_df['std_bac'],
                   marker='o', markersize=10,
                   capsize=5, capthick=2,
                   linewidth=2.5, linestyle='-',
                   color='steelblue',
                   label=f'K={K}, CJM with mode loss')

        # Mark best lambda
        if not np.isnan(best_bac):
            ax.axvline(x=best_lambda, color='red', linestyle='--',
                      alpha=0.7, linewidth=2,
                      label=f'Best λ={best_lambda:.2e} (BAC={best_bac:.3f})')
            ax.scatter([best_lambda], [best_bac],
                      color='red', s=200, zorder=5,
                      edgecolors='darkred', linewidth=2)

        # Add reference lines
        random_bac = 1/K
        ax.axhline(y=random_bac, color='gray', linestyle=':',
                  alpha=0.5, linewidth=1.5,
                  label=f'Random baseline ({random_bac:.3f})')

        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Lambda (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Balanced Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'BAC vs Lambda - K={K} CJM with Mode Loss\n' +
                    f'(7 points, {n_sims} simulations, Mac Air M4 Optimized)',
                    fontsize=15, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)

        # Set y-axis limits for better visibility
        ax.set_ylim([max(0, random_bac - 0.1), min(1.0, valid_df['mean_bac'].max() + 0.05)])

        # Add lambda values as x-tick labels
        ax.set_xticks(valid_df['lambda'].values)
        ax.set_xticklabels([f'{x:.0e}' if x >= 1 else f'{x:.2f}'
                            for x in valid_df['lambda'].values],
                           rotation=45, ha='right')

    plt.tight_layout()
    plot_path = output_dir / 'bac_vs_lambda_7points.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"📈 Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()