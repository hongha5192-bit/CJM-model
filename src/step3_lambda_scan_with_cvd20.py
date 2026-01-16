"""
Lambda scan for K=3 CJM with CVD_20 feature included
Test lambda values: 1, 5, 10, 20
Compare with results without CVD_20
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from itertools import permutations
from jumpmodels.jump import JumpModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fit_cjm_on_simulation(sim_data, lambda_val, K, grid_size, config):
    """
    Fit CJM model on a single simulation with CVD_20 feature
    """
    try:
        # Extract features INCLUDING CVD_20
        feature_names = ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20', 'CVD_20']
        X_raw = sim_data[feature_names].values

        # CRITICAL: Standardize ALL features including CVD_20
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Log CVD_20 standardization info
        cvd_idx = feature_names.index('CVD_20')
        logger.debug(f"CVD_20 - Raw mean: {X_raw[:, cvd_idx].mean():.2f}, Raw std: {X_raw[:, cvd_idx].std():.2f}")
        logger.debug(f"CVD_20 - Standardized mean: {X[:, cvd_idx].mean():.4f}, Standardized std: {X[:, cvd_idx].std():.4f}")

        # Get returns and true states
        ret_ser = sim_data['returns'].values
        s_true = sim_data['true_state'].values

        # Fit CJM using JumpModel
        model = JumpModel(
            n_components=K,
            jump_penalty=lambda_val,
            mode_loss=True,
            grid_size=grid_size,
            n_init=config['lambda_scan']['jumpmodels'].get('n_init', 3),
            max_iter=500,  # Increased for better convergence with more features
            tol=float(config['lambda_scan']['jumpmodels'].get('tol', 1e-5)),
            random_state=123,
            verbose=0
        )

        # Fit with returns for sorting
        model.fit(X, ret_ser=ret_ser, sort_by='cumret')

        # Get predictions
        s_pred = model.predict(X)

        # Calculate permutation-invariant BAC
        bac = calculate_permutation_invariant_bac(s_true, s_pred, K)

        return bac

    except Exception as e:
        logger.error(f"Error fitting simulation: {type(e).__name__}: {e}")
        return np.nan

def calculate_permutation_invariant_bac(y_true, y_pred, K):
    """Calculate best BAC across all permutations"""
    best_bac = 0
    for perm in permutations(range(K)):
        y_pred_perm = np.array([perm[y] for y in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_perm)
        best_bac = max(best_bac, bac)
    return best_bac

def compare_with_baseline():
    """Load and compare with results without CVD_20"""
    baseline_path = Path('outputs/step3_lambda/lambda_scores_K3_with_ursi_breadth.csv')

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        logger.info("\n" + "="*80)
        logger.info("COMPARISON WITH BASELINE (8 features, no CVD_20)")
        logger.info("="*80)

        for idx, row in baseline_df.iterrows():
            logger.info(f"λ={row['lambda']:>3}: BAC={row['mean_bac']:.4f} ± {row['std_bac']:.4f}")

        return baseline_df
    else:
        logger.warning("Baseline results not found")
        return None

def main():
    logger.info("="*80)
    logger.info("LAMBDA SCAN WITH CVD_20 FEATURE (9 Features Total)")
    logger.info("="*80)

    # Load config
    with open('configs/daily_K3.yaml', 'r') as f:
        config = yaml.safe_load(f)

    K = config['hmm']['K']
    grid_size = config['lambda_scan']['cjm']['grid_size']

    # Test lambda values
    lambda_values = [1, 5, 10, 20]
    n_simulations = 50  # Use 50 simulations for robust results

    logger.info(f"\nTesting lambda values: {lambda_values}")
    logger.info(f"Number of simulations: {n_simulations}")
    logger.info("Features: ADX, DMI_Plus, DMI_Minus, URSI, BB_PCTB, BBWP, URSI_0_50, URSI_0_20, CVD_20")

    # Check if simulations exist, if not generate them
    sim_dir = Path('outputs/step2_sim/K3_with_cvd')
    if not sim_dir.exists() or len(list(sim_dir.glob('sim_*.parquet'))) < n_simulations:
        logger.info("\nGenerating simulations with CVD_20...")
        import subprocess
        result = subprocess.run(['python', 'src/step2_simulate_with_cvd.py'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to generate simulations")
            logger.error(result.stderr)
            return

    # Load simulations
    sim_files = sorted(sim_dir.glob('sim_*.parquet'))[:n_simulations]

    if not sim_files:
        raise ValueError(f"No simulation files found in {sim_dir}")

    logger.info(f"\nLoaded {len(sim_files)} simulations from {sim_dir}")

    # Load simulations into memory
    simulations = []
    for sim_file in sim_files:
        sim_df = pd.read_parquet(sim_file)
        simulations.append(sim_df)

    # Run lambda scan
    results = []

    for lambda_val in lambda_values:
        logger.info(f"\nProcessing λ={lambda_val}...")

        bac_scores = []

        for sim_idx, sim_data in enumerate(tqdm(simulations, desc=f"λ={lambda_val}")):
            bac = fit_cjm_on_simulation(sim_data, lambda_val, K, grid_size, config)

            if not np.isnan(bac):
                bac_scores.append(bac)

        if bac_scores:
            mean_bac = np.mean(bac_scores)
            std_bac = np.std(bac_scores)
            n_success = len(bac_scores)

            results.append({
                'lambda': lambda_val,
                'mean_bac': mean_bac,
                'std_bac': std_bac,
                'n_success': n_success,
                'n_total': n_simulations
            })

            logger.info(f"  λ={lambda_val}: BAC={mean_bac:.4f} ± {std_bac:.4f} ({n_success}/{n_simulations} successful)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path('outputs/step3_lambda')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / 'lambda_scores_K3_with_cvd20.csv'
    results_df.to_csv(results_csv, index=False)

    # Find best lambda
    if len(results_df) > 0:
        best_idx = results_df['mean_bac'].idxmax()
        best_lambda = results_df.loc[best_idx, 'lambda']
        best_bac = results_df.loc[best_idx, 'mean_bac']
        best_std = results_df.loc[best_idx, 'std_bac']

        # Save best lambda
        best_params = {
            'lambda': float(best_lambda),
            'mean_bac': float(best_bac),
            'std_bac': float(best_std),
            'features': ['ADX', 'DMI_Plus', 'DMI_Minus', 'URSI', 'BB_PCTB', 'BBWP', 'URSI_0_50', 'URSI_0_20', 'CVD_20'],
            'n_features': 9,
            'n_simulations': n_simulations
        }

        best_json = output_dir / 'best_lambda_K3_with_cvd20.json'
        with open(best_json, 'w') as f:
            json.dump(best_params, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("RESULTS WITH CVD_20 (9 Features)")
        logger.info("="*80)

        for _, row in results_df.iterrows():
            is_best = " ← BEST" if row['lambda'] == best_lambda else ""
            logger.info(f"λ={row['lambda']:>3}: BAC={row['mean_bac']:.4f} ± {row['std_bac']:.4f}{is_best}")

        # Compare with baseline
        baseline_df = compare_with_baseline()

        if baseline_df is not None:
            logger.info("\n" + "="*80)
            logger.info("IMPROVEMENT ANALYSIS")
            logger.info("="*80)

            # Find matching lambda values
            for lambda_val in lambda_values:
                cvd_row = results_df[results_df['lambda'] == lambda_val]
                baseline_row = baseline_df[baseline_df['lambda'] == lambda_val]

                if len(cvd_row) > 0 and len(baseline_row) > 0:
                    cvd_bac = cvd_row['mean_bac'].values[0]
                    baseline_bac = baseline_row['mean_bac'].values[0]
                    improvement = (cvd_bac - baseline_bac) / baseline_bac * 100

                    logger.info(f"λ={lambda_val:>3}: {baseline_bac:.4f} → {cvd_bac:.4f} "
                              f"({'+' if improvement > 0 else ''}{improvement:.1f}%)")

        logger.info("\n" + "="*80)
        logger.info(f"Best λ={best_lambda} with BAC={best_bac:.4f} ± {best_std:.4f}")
        logger.info("="*80)

    else:
        logger.error("No successful fits")

    return results_df

if __name__ == "__main__":
    results = main()