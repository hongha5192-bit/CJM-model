#!/usr/bin/env python
"""
Step 3: Empirical Lambda Scan for K=3 Feature-Based Simulations
Mac Air M4 Optimized: 11 lambdas, 128 simulations, reduced iterations
Uses actual CJM fitting with JumpModel from jumpmodels package
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
import time
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


def fit_cjm_to_simulation(sim_data, lam, K, grid_size, n_init, max_iter, tol, feature_names):
    """
    Fit CJM to a single simulation using features

    Args:
        sim_data: DataFrame with simulation data
        lam: Lambda value
        K: Number of states
        grid_size: Grid size for CJM
        n_init: Number of initializations
        max_iter: Max iterations
        tol: Tolerance
        feature_names: List of feature column names

    Returns:
        BAC score or NaN if failed
    """
    try:
        # Extract features and true labels
        X = sim_data[feature_names].values
        s_true = sim_data['s_true'].values

        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

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
            random_state=42,
            verbose=0
        )

        # Fit the model
        cjm.fit(X_scaled)

        # Get predictions
        s_pred = cjm.predict(X_scaled)

        # Compute BAC with permutation fix
        bac = best_balanced_accuracy_K3(s_true, s_pred)

        return bac

    except Exception as e:
        logger.debug(f"Failed to fit CJM with lambda={lam}: {e}")
        return np.nan


def main():
    """Main function to run empirical lambda scan"""

    # Load configuration
    config_path = 'configs/daily_K3.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']

    # Mac Air M4 optimized parameters
    lambda_grid = np.logspace(-2, 5, 11)  # 11 points from 0.01 to 100,000
    grid_size = 0.05  # Paper recommended for K=3
    n_init = 3  # Reduced for speed
    max_iter = 200  # Reduced for speed
    tol = 1e-5  # Relaxed for speed
    n_simulations = 50  # Use 50 for quick scan (can increase to 128 later)

    logger.info("="*80)
    logger.info("EMPIRICAL LAMBDA SCAN FOR K=3 (MAC AIR M4 OPTIMIZED)")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Lambda grid: {len(lambda_grid)} points")
    logger.info(f"  Lambda range: [{lambda_grid[0]:.2e}, {lambda_grid[-1]:.2e}]")
    logger.info(f"  Grid size = {grid_size}")
    logger.info(f"  n_init = {n_init}, max_iter = {max_iter}, tol = {tol}")
    logger.info(f"  Using {n_simulations} simulations")

    # Load simulations into memory (Mac Air M4 optimization)
    sim_dir = Path('outputs/step2_sim/K3_features')
    sim_files = sorted(sim_dir.glob('sim_T*.parquet'))[:n_simulations]

    if not sim_files:
        logger.error(f"No simulation files found in {sim_dir}")
        return

    logger.info(f"\nLoading {len(sim_files)} simulations into memory...")
    simulations = []
    for sim_file in tqdm(sim_files, desc="Loading"):
        df = pd.read_parquet(sim_file)
        simulations.append(df)

    total_size_mb = sum(df.memory_usage(deep=True).sum() for df in simulations) / 1e6
    logger.info(f"Loaded {len(simulations)} simulations ({total_size_mb:.1f} MB)")

    # Lambda scan
    results = []
    logger.info("\nStarting lambda scan...")

    start_time = time.time()

    for lambda_idx, lam in enumerate(lambda_grid):
        logger.info(f"\nProcessing λ = {lam:.2e} ({lambda_idx+1}/{len(lambda_grid)})...")

        bac_scores = []

        # Process each simulation
        for sim_idx, sim_data in enumerate(tqdm(simulations,
                                                desc=f"λ={lam:.2e}",
                                                leave=False)):
            bac = fit_cjm_to_simulation(
                sim_data, lam, K, grid_size,
                n_init, max_iter, tol, feature_names
            )

            if not np.isnan(bac):
                bac_scores.append(bac)

        # Calculate statistics
        if bac_scores:
            mean_bac = np.mean(bac_scores)
            std_bac = np.std(bac_scores)
            n_valid = len(bac_scores)
        else:
            mean_bac = np.nan
            std_bac = np.nan
            n_valid = 0

        results.append({
            'lambda': lam,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': n_valid,
            'n_total': len(simulations)
        })

        logger.info(f"  Result: BAC = {mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(simulations)} valid)")

    elapsed = time.time() - start_time
    logger.info(f"\nLambda scan completed in {elapsed:.1f} seconds")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Find best lambda
    valid_results = df_results[~df_results['mean_bac'].isna()]

    if len(valid_results) > 0:
        best_idx = valid_results['mean_bac'].idxmax()
        best_lambda = valid_results.loc[best_idx, 'lambda']
        best_bac = valid_results.loc[best_idx, 'mean_bac']
        best_std = valid_results.loc[best_idx, 'std_bac']

        logger.info("\n" + "="*80)
        logger.info("LAMBDA SELECTION RESULTS")
        logger.info("="*80)
        logger.info(f"Best lambda: {best_lambda:.2e}")
        logger.info(f"Best mean BAC: {best_bac:.4f} ± {best_std:.4f}")

        # Save results
        output_dir = Path('outputs/step3_lambda')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scores CSV
        scores_path = output_dir / 'lambda_scores_K3_empirical.csv'
        df_results.to_csv(scores_path, index=False)
        logger.info(f"\nScores saved to: {scores_path}")

        # Save best lambda JSON
        best_lambda_data = {
            'lambda': float(best_lambda),
            'mean_bac': float(best_bac),
            'std_bac': float(best_std),
            'K': K,
            'grid_size': grid_size,
            'n_simulations': len(simulations),
            'model_type': 'CJM_mode',
            'features': feature_names,
            'method': 'empirical_mac_air_m4_optimized'
        }

        best_path = output_dir / 'best_lambda_K3_empirical.json'
        with open(best_path, 'w') as f:
            json.dump(best_lambda_data, f, indent=2)
        logger.info(f"Best lambda saved to: {best_path}")

        # Create visualization
        create_lambda_plot(df_results, best_lambda, output_dir)

        # Display results table
        logger.info("\n" + "="*80)
        logger.info("FULL RESULTS TABLE")
        logger.info("="*80)
        print(df_results.to_string(index=False))

    else:
        logger.error("All lambda values failed!")

    logger.info("\n✅ Empirical lambda scan completed successfully!")


def create_lambda_plot(df_results, best_lambda, output_dir):
    """Create BAC vs Lambda plot"""

    # Filter valid results
    df_valid = df_results[~df_results['mean_bac'].isna()].copy()

    if len(df_valid) == 0:
        return

    plt.figure(figsize=(10, 6))

    # Plot mean BAC with error bars
    plt.errorbar(df_valid['lambda'], df_valid['mean_bac'],
                yerr=df_valid['std_bac'],
                marker='o', markersize=8,
                linewidth=2, capsize=5,
                label='Mean BAC ± σ')

    # Mark best lambda
    best_row = df_valid[df_valid['lambda'] == best_lambda].iloc[0]
    plt.plot(best_lambda, best_row['mean_bac'],
            'r*', markersize=20,
            label=f'Best λ={best_lambda:.2e}')

    # Add horizontal line at random performance
    plt.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5,
                label='Random (BAC=0.333)')

    plt.xscale('log')
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Mean Balanced Accuracy', fontsize=12)
    plt.title('Empirical Lambda Selection: K=3 CJM with Features', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / 'bac_vs_lambda_K3_empirical.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()