#!/usr/bin/env python
"""
Step 3: Lambda scan with custom values using corrected simulations
Tests lambdas: 5, 10, 15, 20, 25, 30, 50, 100
Uses 50 simulations with max_iter=500
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

        # Fit CJM with mode loss - Custom settings
        cjm = JumpModel(
            n_components=K,
            jump_penalty=lam,
            cont=True,              # Continuous model
            mode_loss=True,         # Mode loss version
            grid_size=grid_size,    # 0.05 for K=3
            n_init=3,               # Slightly increased for better initialization
            max_iter=500,           # As requested
            tol=1e-6,              # Tighter tolerance with more iterations
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


def process_lambda(lam, sim_files, K, feature_names, grid_size, n_jobs=2):
    """Process a single lambda value"""

    logger.info(f"\nProcessing λ = {lam}")

    # Process simulations for this lambda with progress bar
    with tqdm(total=len(sim_files), desc=f"λ={lam}", unit="sim", leave=False) as pbar:
        bac_scores = []
        for sim_path in sim_files:
            bac = fit_cjm_single(sim_path, lam, K, feature_names, grid_size)
            bac_scores.append(bac)
            pbar.update(1)

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

    # Log result
    if not np.isnan(mean_bac):
        logger.info(f"λ={lam:6.0f}: BAC={mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(sim_files)} valid)")
    else:
        logger.info(f"λ={lam:6.0f}: BAC=NaN (all failed)")

    return result


def main():
    """Run lambda scan with custom values"""

    logger.info("="*80)
    logger.info("LAMBDA SCAN - CUSTOM VALUES")
    logger.info("="*80)

    # Load configuration
    config_path = Path('configs/daily_K3.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']
    grid_size = 0.05  # For K=3 as per paper

    # Custom lambda grid as specified
    lambda_grid = [5, 10, 15, 20, 25, 30, 50, 100]

    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Lambda values: {lambda_grid}")
    logger.info(f"  Number of lambdas: {len(lambda_grid)}")
    logger.info(f"  Grid size = {grid_size}")
    logger.info(f"  CJM settings: n_init=3, max_iter=500, tol=1e-6")

    # Load corrected simulations - use 100 simulations for robust results
    sim_dir = Path('outputs/step2_sim/K3_512_corrected')
    all_sim_files = sorted(sim_dir.glob('sim_*.parquet'))

    # Use 100 simulations for very reliable results
    n_sims_to_use = 100
    sim_files = all_sim_files[:n_sims_to_use]

    if not sim_files:
        logger.error(f"No simulation files found in {sim_dir}")
        return

    logger.info(f"Using {len(sim_files)} simulations (as requested)")

    # Verify first simulation has correct ranges
    df_test = pd.read_parquet(sim_files[0])
    logger.info("\nVerifying simulation data ranges:")
    for feat in feature_names:
        logger.info(f"  {feat:10s}: mean={df_test[feat].mean():6.2f}, std={df_test[feat].std():6.2f}, "
                   f"min={df_test[feat].min():6.2f}, max={df_test[feat].max():6.2f}")

    # Calculate total fits
    total_fits = len(lambda_grid) * len(sim_files)
    logger.info(f"\nTotal CJM fits to perform: {total_fits:,}")
    logger.info(f"Estimated time: {total_fits * 0.5 / 60:.1f} minutes (at ~0.5s per fit with max_iter=500)")

    results = []
    start_time = time.time()

    logger.info("\nStarting lambda scan...")
    logger.info("-" * 60)

    # Process each lambda value sequentially
    for i, lam in enumerate(lambda_grid):
        logger.info(f"\n[{i+1}/{len(lambda_grid)}] Testing λ = {lam}")
        lambda_start = time.time()

        result = process_lambda(lam, sim_files, K, feature_names, grid_size)
        results.append(result)

        lambda_time = time.time() - lambda_start
        logger.info(f"  Time for λ={lam}: {lambda_time:.1f} seconds")

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
        logger.info(f"\n🎯 Best Lambda: {best_lambda:.0f}")
        logger.info(f"   Best Mean BAC: {best_bac:.4f} ± {best_std:.4f}")

    # Display all lambda results
    logger.info("\nAll Lambda Values (in test order):")
    logger.info("-" * 70)
    logger.info(f"{'Lambda':>8} | {'Mean BAC':>10} | {'Std BAC':>10} | {'Valid':>7} | {'Rank':>6}")
    logger.info("-" * 70)

    # Add rank to results
    df_results['rank'] = df_results['mean_bac'].rank(ascending=False, method='min')

    for _, row in df_results.iterrows():
        if not np.isnan(row['mean_bac']):
            rank_str = f"#{int(row['rank'])}" if not np.isnan(row['rank']) else "N/A"
            logger.info(f"{row['lambda']:8.0f} | {row['mean_bac']:10.4f} | "
                       f"{row['std_bac']:10.4f} | {int(row['n_valid']):4d}/{int(row['n_total']):<3d} | {rank_str:>6s}")

    # Save results
    output_dir = Path('outputs/step3_lambda')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save lambda scores CSV
    csv_path = output_dir / 'lambda_scores_custom.csv'
    df_results['K'] = K
    df_results['model_type'] = 'CJM_mode'
    df_results['T'] = 1000
    df_results['N_sim'] = len(sim_files)
    df_results['max_iter'] = 500
    df_results = df_results[['K', 'model_type', 'lambda', 'T', 'N_sim', 'max_iter',
                             'mean_bac', 'std_bac', 'n_valid', 'n_total', 'rank']]
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
                "max_iterations": 500,
                "lambda_values_tested": lambda_grid,
                "simulation_dir": str(sim_dir)
            }
        }
    }
    json_path = output_dir / 'best_lambda_custom.json'
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

    # Show top 3 lambdas
    logger.info("\nTop 3 Lambda Values by BAC:")
    df_top = df_results.nsmallest(3, 'rank')
    for _, row in df_top.iterrows():
        if not np.isnan(row['mean_bac']):
            logger.info(f"  λ={row['lambda']:3.0f}: BAC={row['mean_bac']:.4f} ± {row['std_bac']:.4f}")

    return best_lambda, best_bac


def create_bac_plot(df_results, best_lambda, best_bac, K, output_dir, n_sims):
    """Create BAC vs Lambda plot"""

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Filter valid points
    valid_df = df_results[~df_results['mean_bac'].isna()].copy()
    valid_df = valid_df.sort_values('lambda')  # Sort for line plot

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
                      label=f'Best λ={best_lambda:.0f} (BAC={best_bac:.3f})')
            ax.scatter([best_lambda], [best_bac],
                      color='red', s=200, zorder=5,
                      edgecolors='darkred', linewidth=2)

        # Add reference lines
        random_bac = 1/K
        ax.axhline(y=random_bac, color='gray', linestyle=':',
                  alpha=0.5, linewidth=1.5,
                  label=f'Random baseline ({random_bac:.3f})')

        # Formatting
        ax.set_xlabel('Lambda (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Balanced Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(f'BAC vs Lambda - K={K} CJM with Mode Loss\n' +
                    f'(Custom λ values: 5-100, {n_sims} simulations, max_iter=500)',
                    fontsize=15, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)

        # Set y-axis limits for better visibility
        ax.set_ylim([max(0, random_bac - 0.1), min(1.0, valid_df['mean_bac'].max() + 0.05)])

        # Set x-axis to show all lambda values clearly
        ax.set_xticks(valid_df['lambda'].values)
        ax.set_xticklabels([f'{int(x)}' for x in valid_df['lambda'].values])

        # Add value labels on points
        for _, row in valid_df.iterrows():
            ax.annotate(f'{row["mean_bac"]:.3f}',
                       xy=(row['lambda'], row['mean_bac']),
                       xytext=(0, 5), textcoords='offset points',
                       fontsize=9, ha='center')

    plt.tight_layout()
    plot_path = output_dir / 'bac_vs_lambda_custom.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"📈 Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()