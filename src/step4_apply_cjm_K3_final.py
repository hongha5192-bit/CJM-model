#!/usr/bin/env python
"""
Step 4: Apply CJM K=3 to Real VNINDEX Data
Uses empirically selected lambda=10 with 6 features
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Apply CJM K=3 to real VNINDEX data"""

    # Load configuration
    config_path = 'configs/daily_K3.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = 3
    feature_names = config['features']['cols']

    # Load best lambda from empirical selection
    best_lambda_path = Path('outputs/step3_lambda/best_lambda_K3_final.json')
    if best_lambda_path.exists():
        with open(best_lambda_path, 'r') as f:
            lambda_data = json.load(f)
            # Use production lambda (10) instead of overfitted value
            lambda_star = 10.0  # Based on empirical analysis
    else:
        lambda_star = 10.0  # Default based on empirical study

    logger.info("="*80)
    logger.info("APPLYING CJM K=3 TO REAL VNINDEX DATA")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  K = {K}")
    logger.info(f"  Lambda = {lambda_star}")
    logger.info(f"  Features = {feature_names}")
    logger.info(f"  Grid size = 0.05")

    # Load prepared data
    data_path = Path('outputs/prepared_data.csv')
    if not data_path.exists():
        logger.error(f"Prepared data not found at {data_path}")
        logger.info("Please run step0_prepare_vnindex.py first")
        return

    df = pd.read_csv(data_path, parse_dates=['date'])
    logger.info(f"Loaded {len(df)} days of data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Extract features and handle missing values
    X = df[feature_names].values

    # Check for missing values
    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        logger.warning(f"Found {n_missing} missing values, will forward-fill")
        df[feature_names] = df[feature_names].fillna(method='ffill')
        X = df[feature_names].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("\nFeature statistics (before scaling):")
    for i, feat in enumerate(feature_names):
        logger.info(f"  {feat:10s}: mean={X[:, i].mean():7.2f}, std={X[:, i].std():7.2f}")

    # Fit CJM with empirically selected lambda
    logger.info(f"\nFitting CJM with K={K}, lambda={lambda_star}...")

    cjm = JumpModel(
        n_components=K,
        jump_penalty=lambda_star,
        cont=True,
        mode_loss=True,
        grid_size=0.05,  # K=3 optimized
        n_init=10,  # Higher for real data robustness
        max_iter=1000,
        tol=1e-8,
        random_state=42,
        verbose=1
    )

    # Fit the model
    cjm.fit(X_scaled)

    # Get predictions
    regimes = cjm.predict(X_scaled)
    proba = cjm.predict_proba(X_scaled)

    # Add results to dataframe
    df['regime'] = regimes
    for k in range(K):
        df[f'proba_{k}'] = proba[:, k]

    # Analyze regime characteristics
    logger.info("\n" + "="*80)
    logger.info("REGIME ANALYSIS")
    logger.info("="*80)

    # Regime distribution
    regime_counts = df['regime'].value_counts().sort_index()
    logger.info("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  Regime {regime}: {count:4d} days ({pct:5.1f}%)")

    # Regime statistics
    logger.info("\nRegime Characteristics:")
    for regime in range(K):
        regime_data = df[df['regime'] == regime]
        if len(regime_data) > 0:
            ret_mean = regime_data['ret'].mean() * 100
            ret_std = regime_data['ret'].std() * 100
            logger.info(f"\n  Regime {regime}:")
            logger.info(f"    Days: {len(regime_data)}")
            logger.info(f"    Return: {ret_mean:+.3f}% (daily avg)")
            logger.info(f"    Volatility: {ret_std:.3f}% (daily std)")

            # Feature means in this regime
            for feat in feature_names:
                feat_mean = regime_data[feat].mean()
                logger.info(f"    {feat}: {feat_mean:.2f}")

    # Calculate regime switches
    switches = (df['regime'].diff() != 0).sum() - 1  # Subtract 1 for first NaN
    switches_per_year = switches / (len(df) / 252)
    logger.info(f"\nRegime Dynamics:")
    logger.info(f"  Total switches: {switches}")
    logger.info(f"  Switches per year: {switches_per_year:.1f}")
    logger.info(f"  Average regime duration: {len(df)/switches:.1f} days")

    # Save results
    output_dir = Path('outputs/step4_apply')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save regime predictions
    output_cols = ['date', 'close', 'ret', 'regime'] + [f'proba_{k}' for k in range(K)]
    output_df = df[output_cols].copy()

    output_path = output_dir / 'regimes_daily_K3.csv'
    output_df.to_csv(output_path, index=False)
    logger.info(f"\nRegime predictions saved to: {output_path}")

    # Save model parameters
    model_params = {
        'K': K,
        'lambda': lambda_star,
        'features': feature_names,
        'grid_size': 0.05,
        'n_observations': len(df),
        'date_range': {
            'start': str(df['date'].min()),
            'end': str(df['date'].max())
        },
        'regime_distribution': regime_counts.to_dict(),
        'n_switches': int(switches),
        'switches_per_year': float(switches_per_year),
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        }
    }

    params_path = output_dir / 'cjm_K3_params.json'
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=2)
    logger.info(f"Model parameters saved to: {params_path}")

    # Create visualizations
    create_regime_plot(df, output_dir)
    create_transition_heatmap(df, K, output_dir)

    logger.info("\n" + "="*80)
    logger.info("✅ CJM APPLICATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Generated files:")
    logger.info(f"  - Regime predictions: {output_path}")
    logger.info(f"  - Model parameters: {params_path}")
    logger.info(f"  - Visualizations in: {output_dir}")

    return df


def create_regime_plot(df, output_dir):
    """Create regime visualization over time"""

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Plot 1: Price with regime colors
    ax1 = axes[0]
    colors = {0: 'green', 1: 'gray', 2: 'red'}

    for regime in range(3):
        mask = df['regime'] == regime
        if mask.any():
            ax1.scatter(df.loc[mask, 'date'], df.loc[mask, 'close'],
                       c=colors[regime], alpha=0.6, s=5,
                       label=f'Regime {regime}')

    ax1.plot(df['date'], df['close'], 'k-', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('VNINDEX Close')
    ax1.set_title('VNINDEX with CJM K=3 Regime Classification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Regime indicator
    ax2 = axes[1]
    ax2.plot(df['date'], df['regime'], 'b-', linewidth=1)
    ax2.fill_between(df['date'], 0, df['regime'], alpha=0.3)
    ax2.set_ylabel('Regime')
    ax2.set_ylim(-0.1, 2.1)
    ax2.set_yticks([0, 1, 2])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Regime probabilities
    ax3 = axes[2]
    ax3.plot(df['date'], df['proba_0'], 'g-', label='P(Regime 0)', alpha=0.7)
    ax3.plot(df['date'], df['proba_1'], 'gray', label='P(Regime 1)', alpha=0.7)
    ax3.plot(df['date'], df['proba_2'], 'r-', label='P(Regime 2)', alpha=0.7)
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'regime_timeline_K3.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"  - Regime timeline plot: {plot_path}")


def create_transition_heatmap(df, K, output_dir):
    """Create transition probability heatmap"""

    # Calculate empirical transition matrix
    transitions = np.zeros((K, K))

    for i in range(len(df) - 1):
        from_state = df.iloc[i]['regime']
        to_state = df.iloc[i + 1]['regime']
        transitions[from_state, to_state] += 1

    # Normalize to probabilities
    for i in range(K):
        if transitions[i].sum() > 0:
            transitions[i] /= transitions[i].sum()

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(transitions, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0, vmax=1, square=True,
                xticklabels=[f'To {i}' for i in range(K)],
                yticklabels=[f'From {i}' for i in range(K)])
    plt.title('Empirical Regime Transition Probabilities')

    plot_path = output_dir / 'transition_matrix_K3.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    logger.info(f"  - Transition matrix plot: {plot_path}")


if __name__ == '__main__':
    main()