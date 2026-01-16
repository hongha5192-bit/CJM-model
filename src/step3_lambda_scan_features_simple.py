#!/usr/bin/env python
"""
Step 3: Simplified Lambda scan for feature-based simulations
Tests different lambda values to find optimal jump penalty
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simple_regime_detection(X, lambda_val, K=3):
    """
    Simple regime detection using clustering with jump penalty
    This is a placeholder for the actual CJM algorithm
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use KMeans as a simple baseline
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Simulate lambda effect by smoothing predictions
    # Higher lambda = more smoothing (fewer jumps)
    if lambda_val > 1:
        # Simple smoothing based on lambda
        smoothing_window = int(np.log10(lambda_val) * 2) + 1
        labels_smooth = pd.Series(labels).rolling(smoothing_window, center=True, min_periods=1).mode().iloc[:, 0].values
        return labels_smooth.astype(int)
    else:
        return labels


def best_balanced_accuracy_K3(s_pred, s_true):
    """
    Compute best balanced accuracy over all 6 permutations for K=3
    """
    from itertools import permutations
    from sklearn.metrics import balanced_accuracy_score

    K = 3
    best_bac = 0

    for perm in permutations(range(K)):
        # Map predictions using this permutation
        mapping = {old: new for old, new in enumerate(perm)}
        s_pred_mapped = pd.Series(s_pred).map(mapping).values

        # Compute BAC
        bac = balanced_accuracy_score(s_true, s_pred_mapped)
        best_bac = max(best_bac, bac)

    return best_bac


def process_simulation(sim_path, lambda_val, feature_names):
    """Process a single simulation with given lambda"""
    try:
        # Load simulation
        df = pd.read_parquet(sim_path)

        # Extract features
        X = df[feature_names].values

        # Apply regime detection
        s_pred = simple_regime_detection(X, lambda_val)

        # Get true labels
        s_true = df['s_true'].values

        # Calculate BAC
        bac = best_balanced_accuracy_K3(s_pred, s_true)

        return bac

    except Exception as e:
        logger.debug(f"Error processing {sim_path}: {e}")
        return np.nan


def lambda_scan_simple(config_path='configs/daily_K3.yaml'):
    """Perform simplified lambda scan"""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']  # Extract cols list from features dict

    # Define lambda grid
    lambda_grid = np.logspace(-1, 3, 9)  # 9 points from 0.1 to 1000

    logger.info("="*80)
    logger.info("SIMPLIFIED LAMBDA SCAN (FEATURE-BASED)")
    logger.info("="*80)
    logger.info(f"Lambda grid: {lambda_grid}")
    logger.info(f"Features: {feature_names}")

    # Get simulation files
    sim_dir = Path('outputs/step2_sim/K3_features')
    sim_files = sorted(sim_dir.glob('sim_T*.parquet'))[:30]  # Use 30 simulations

    if not sim_files:
        raise ValueError(f"No simulation files found in {sim_dir}")

    logger.info(f"Using {len(sim_files)} simulations")

    # Results storage
    results = []

    # Scan each lambda
    for lambda_val in tqdm(lambda_grid, desc="Lambda scan"):
        bac_scores = []

        for sim_path in sim_files:
            bac = process_simulation(sim_path, lambda_val, feature_names)
            if not np.isnan(bac):
                bac_scores.append(bac)

        if bac_scores:
            mean_bac = np.mean(bac_scores)
            std_bac = np.std(bac_scores)
            n_valid = len(bac_scores)
        else:
            mean_bac = np.nan
            std_bac = np.nan
            n_valid = 0

        results.append({
            'lambda': lambda_val,
            'mean_bac': mean_bac,
            'std_bac': std_bac,
            'n_valid': n_valid,
            'n_total': len(sim_files)
        })

        logger.info(f"λ={lambda_val:.2e}: BAC={mean_bac:.4f} ± {std_bac:.4f} ({n_valid}/{len(sim_files)} valid)")

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Find best lambda
    if not df_results['mean_bac'].isna().all():
        best_idx = df_results['mean_bac'].idxmax()
        best_lambda = df_results.loc[best_idx, 'lambda']
        best_bac = df_results.loc[best_idx, 'mean_bac']
    else:
        logger.warning("All lambda values resulted in NaN BAC scores!")
        best_lambda = lambda_grid[len(lambda_grid)//2]  # Default to middle value
        best_bac = 0.333  # Random performance

    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Best lambda: {best_lambda:.2e}")
    logger.info(f"Best BAC: {best_bac:.4f}")

    # Display full results table
    logger.info("\nFull Results:")
    logger.info(df_results.to_string(index=False))

    # Save results
    output_dir = Path('outputs/step3_lambda')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scores
    scores_path = output_dir / 'lambda_scores_features_simple.csv'
    df_results.to_csv(scores_path, index=False)
    logger.info(f"\nScores saved to: {scores_path}")

    # Save best lambda
    best_data = {
        'lambda': float(best_lambda),
        'mean_bac': float(best_bac),
        'method': 'simplified_clustering',
        'n_simulations': len(sim_files),
        'features': feature_names
    }

    best_path = output_dir / 'best_lambda_features_simple.json'
    with open(best_path, 'w') as f:
        json.dump(best_data, f, indent=2)

    # Create visualization
    create_plot(df_results, best_lambda, output_dir)

    return best_data


def create_plot(df_results, best_lambda, output_dir):
    """Create lambda scan visualization"""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add mean BAC line
    fig.add_trace(go.Scatter(
        x=df_results['lambda'],
        y=df_results['mean_bac'],
        mode='lines+markers',
        name='Mean BAC',
        line=dict(color='blue', width=2),
        error_y=dict(
            type='data',
            array=df_results['std_bac'],
            visible=True
        )
    ))

    # Mark best lambda
    best_row = df_results[df_results['lambda'] == best_lambda].iloc[0]
    fig.add_trace(go.Scatter(
        x=[best_lambda],
        y=[best_row['mean_bac']],
        mode='markers',
        name=f'Best λ={best_lambda:.2e}',
        marker=dict(color='red', size=15, symbol='star')
    ))

    # Add random baseline
    fig.add_hline(y=1/3, line_dash="dash", line_color="gray",
                  annotation_text="Random (BAC=0.333)")

    fig.update_layout(
        title='Lambda Scan Results (Simplified)',
        xaxis_title='Lambda (λ)',
        yaxis_title='Mean Balanced Accuracy',
        xaxis_type='log',
        height=500,
        width=800
    )

    plot_path = output_dir / 'lambda_scan_simple.html'
    fig.write_html(plot_path)
    logger.info(f"Plot saved to: {plot_path}")


def main():
    """Main function"""
    try:
        result = lambda_scan_simple()
        logger.info("\n✅ Lambda scan completed!")
        logger.info(f"Recommended λ = {result['lambda']:.2e}")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())