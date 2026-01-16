#!/usr/bin/env python
"""
Step 4: Fit CJM on real VNINDEX data with K=3 using selected lambda
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import yaml
from jumpmodels.jump import JumpModel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def fit_cjm_realdata_K3(config_path='configs/daily_K3.yaml'):
    """Fit CJM with K=3 on real VNINDEX data using selected lambda"""

    # Load config
    logger.info("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    K = config['apply']['K']
    assert K == 3, f"Expected K=3, got K={K}"

    # Load best lambda
    best_lambda_path = Path('outputs/step3_lambda/best_lambda_K3.json')
    if not best_lambda_path.exists():
        raise FileNotFoundError(f"Best lambda not found at {best_lambda_path}")

    with open(best_lambda_path, 'r') as f:
        best_lambda_dict = json.load(f)

    lambda_star = best_lambda_dict[f'K{K}']['CJM_mode']
    logger.info(f"Using lambda = {lambda_star:.2e}")

    # Load prepared data
    data_path = Path('outputs/prepared_data.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Prepared data not found at {data_path}")

    df = pd.read_csv(data_path, parse_dates=['date'])
    logger.info(f"Loaded {len(df)} rows of data")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Extract features and returns
    feature_cols = config['features']['cols']
    X_raw = df[feature_cols].values
    ret_ser = df['ret'].values
    dates = df['date'].values

    # Remove NaN rows
    valid_idx = ~np.any(np.isnan(X_raw), axis=1) & ~np.isnan(ret_ser)
    X_raw = X_raw[valid_idx]
    ret_ser = ret_ser[valid_idx]
    dates = dates[valid_idx]

    logger.info(f"Using {len(X_raw)} valid observations")
    logger.info(f"Feature columns: {feature_cols}")

    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_real = scaler.fit_transform(X_raw)

    # CJM settings for real data
    model_type = config['apply']['model_type']
    n_init = config['apply']['n_init']
    max_iter = config['apply']['max_iter']
    tol = config['apply']['tol']
    grid_size = config['lambda_scan']['cjm']['grid_size']  # Critical for K=3

    logger.info(f"CJM settings for real data:")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  K = {K}")
    logger.info(f"  lambda = {lambda_star:.2e}")
    logger.info(f"  grid_size = {grid_size}")
    logger.info(f"  n_init = {n_init}")
    logger.info(f"  max_iter = {max_iter}")
    logger.info(f"  tol = {tol}")

    # Fit CJM with mode loss
    logger.info("Fitting CJM on real data...")
    cjm_real = JumpModel(
        n_components=K,
        jump_penalty=lambda_star,
        cont=True,
        mode_loss=True,
        grid_size=grid_size,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=123,
        verbose=1
    )

    # Fit with return series for sorting (important for K=3 label consistency)
    cjm_real.fit(X_real, ret_ser=ret_ser, sort_by='cumret')

    # Get predictions
    labels = cjm_real.predict(X_real)
    probas = cjm_real.predict_proba(X_real)

    logger.info(f"Fitted CJM with {K} states")

    # Create output dataframe
    output_df = pd.DataFrame({
        'date': dates,
        'ret': ret_ser,
        'label': labels,
        'proba_0': probas[:, 0],
        'proba_1': probas[:, 1],
        'proba_2': probas[:, 2]
    })

    # Sort by date
    output_df = output_df.sort_values('date').reset_index(drop=True)

    # Save main results
    output_path = Path('outputs/step4_apply')
    output_path.mkdir(parents=True, exist_ok=True)

    regimes_path = output_path / 'regimes_daily_K3.csv'
    output_df.to_csv(regimes_path, index=False)
    logger.info(f"Saved regime predictions to {regimes_path}")

    # Extract and save model parameters
    model_params = {
        'K': K,
        'lambda': lambda_star,
        'model_type': model_type,
        'grid_size': grid_size,
        'features': feature_cols,
        'n_observations': len(X_real),
        'date_range': {
            'start': str(output_df['date'].min()),
            'end': str(output_df['date'].max())
        }
    }

    # Get centers (means for each state and feature)
    if hasattr(cjm_real, 'model_') and hasattr(cjm_real.model_, 'means_'):
        centers = cjm_real.model_.means_
        model_params['centers'] = {}
        for k in range(K):
            model_params['centers'][f'state_{k}'] = {}
            for j, feat in enumerate(feature_cols):
                model_params['centers'][f'state_{k}'][feat] = float(centers[k, j])

    # Try to extract transition matrix if available
    if hasattr(cjm_real, 'transmat_'):
        model_params['transition_matrix'] = cjm_real.transmat_.tolist()
    else:
        # Estimate empirical transition matrix
        P_emp = np.zeros((K, K))
        for i in range(len(labels) - 1):
            P_emp[labels[i], labels[i+1]] += 1
        # Normalize rows
        for i in range(K):
            if P_emp[i].sum() > 0:
                P_emp[i] /= P_emp[i].sum()
        model_params['empirical_transition_matrix'] = P_emp.tolist()

    # Save model parameters
    params_path = output_path / 'cjm_model_params_K3.json'
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=2)

    logger.info(f"Saved model parameters to {params_path}")

    # Calculate and print summary statistics
    print("\n" + "="*70)
    print("CJM K=3 FIT SUMMARY ON REAL DATA")
    print("="*70)
    print(f"Data period: {output_df['date'].min().date()} to {output_df['date'].max().date()}")
    print(f"Number of observations: {len(output_df)}")
    print(f"Number of states: {K}")
    print(f"Lambda (jump penalty): {lambda_star:.2e}")
    print(f"Features used: {', '.join(feature_cols)}")

    # Regime statistics
    print("\nRegime Distribution:")
    regime_counts = output_df['label'].value_counts().sort_index()
    regime_pcts = output_df['label'].value_counts(normalize=True).sort_index()
    for state in range(K):
        count = regime_counts.get(state, 0)
        pct = regime_pcts.get(state, 0.0)
        print(f"  State {state}: {count:4d} days ({pct:6.2%})")

    # Return statistics by regime
    print("\nReturn Statistics by Regime:")
    for state in range(K):
        state_data = output_df[output_df['label'] == state]
        if len(state_data) > 0:
            mean_ret = state_data['ret'].mean()
            std_ret = state_data['ret'].std()
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            print(f"  State {state}: mean={mean_ret:7.4f}, vol={std_ret:7.4f}, Sharpe={sharpe:6.2f}")

    # Regime persistence
    print("\nRegime Persistence:")
    switches = (output_df['label'].diff() != 0).sum()
    avg_duration = len(output_df) / (switches + 1)
    print(f"  Total regime switches: {switches}")
    print(f"  Average regime duration: {avg_duration:.1f} days")

    # Calculate average duration per state
    current_state = output_df['label'].iloc[0]
    state_durations = {i: [] for i in range(K)}
    duration = 1

    for i in range(1, len(output_df)):
        if output_df['label'].iloc[i] == current_state:
            duration += 1
        else:
            state_durations[current_state].append(duration)
            current_state = output_df['label'].iloc[i]
            duration = 1

    # Add last duration
    state_durations[current_state].append(duration)

    print("\nAverage Duration by State:")
    for state in range(K):
        durations = state_durations[state]
        if durations:
            avg_dur = np.mean(durations)
            max_dur = np.max(durations)
            print(f"  State {state}: {avg_dur:.1f} days (max: {max_dur} days)")

    # Transition matrix
    if 'empirical_transition_matrix' in model_params:
        print("\nEmpirical Transition Matrix:")
        P = np.array(model_params['empirical_transition_matrix'])
        for i in range(K):
            row = " ".join([f"{p:6.4f}" for p in P[i]])
            print(f"  P[{i}→*] = [{row}]")

    # Feature centers (if available)
    if 'centers' in model_params:
        print("\nFeature Centers by State (standardized):")
        for state in range(K):
            print(f"  State {state}:")
            for feat in feature_cols[:3]:  # Show first 3 features
                val = model_params['centers'][f'state_{state}'][feat]
                print(f"    {feat:10s}: {val:7.3f}")

    print("="*70)

    # Create a summary plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Returns colored by regime
        ax1 = axes[0]
        colors = ['red', 'yellow', 'green']  # Low, medium, high states
        for state in range(K):
            mask = output_df['label'] == state
            ax1.scatter(output_df.loc[mask, 'date'],
                       output_df.loc[mask, 'ret'],
                       c=colors[state], alpha=0.6, s=10,
                       label=f'State {state}')
        ax1.set_ylabel('Daily Return')
        ax1.set_title(f'VNINDEX Returns by Regime (K={K})', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)

        # Plot 2: Regime labels over time
        ax2 = axes[1]
        ax2.plot(output_df['date'], output_df['label'], drawstyle='steps-post', linewidth=1)
        ax2.fill_between(output_df['date'], output_df['label'],
                         step='post', alpha=0.3)
        ax2.set_ylabel('Regime')
        ax2.set_title('Regime Evolution Over Time', fontsize=12)
        ax2.set_ylim(-0.1, K-0.9)
        ax2.set_yticks(range(K))
        ax2.grid(True, alpha=0.3)

        # Plot 3: Probability distributions
        ax3 = axes[2]
        ax3.plot(output_df['date'], output_df['proba_0'], 'r-', alpha=0.7, label='P(State 0)')
        ax3.plot(output_df['date'], output_df['proba_1'], 'y-', alpha=0.7, label='P(State 1)')
        ax3.plot(output_df['date'], output_df['proba_2'], 'g-', alpha=0.7, label='P(State 2)')
        ax3.set_ylabel('Probability')
        ax3.set_xlabel('Date')
        ax3.set_title('Regime Probabilities Over Time', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))

        plt.suptitle(f'CJM K={K} Analysis - VNINDEX (2018-2025)', fontsize=14, y=1.02)
        plt.tight_layout()

        plot_path = output_path / 'regime_analysis_K3.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved analysis plot to {plot_path}")

    except Exception as e:
        logger.warning(f"Could not create plot: {e}")

    return output_df


if __name__ == "__main__":
    try:
        fit_cjm_realdata_K3()
    except Exception as e:
        logger.error(f"Error fitting CJM: {e}", exc_info=True)
        raise