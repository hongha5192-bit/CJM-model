#!/usr/bin/env python3
"""
Display fitted HMM parameters in a clear, formatted manner
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

def display_hmm_parameters():
    """Display all HMM fitted parameters"""

    # Load parameters
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    print("="*80)
    print("FITTED HMM PARAMETERS (K=3)")
    print("="*80)

    # Basic info
    print(f"\nModel Information:")
    print(f"  Number of states (K): {params['K']}")
    print(f"  Number of observations: {params['n_obs']}")
    print(f"  Training period: {params['train_start']} to {params['train_end']}")
    print(f"  Log-likelihood: {params['log_likelihood']:.2f}")
    print(f"  Features: {', '.join(params['features'])}")

    # State interpretation based on features
    state_labels = {
        0: "Bullish (State 0)",
        1: "Neutral (State 1)",
        2: "Bearish (State 2)"
    }

    print("\n" + "="*80)
    print("INITIAL STATE PROBABILITIES (π)")
    print("="*80)
    for i, prob in enumerate(params['pi']):
        print(f"  {state_labels[i]}: {prob:.4f} ({prob*100:.2f}%)")

    print("\n" + "="*80)
    print("TRANSITION MATRIX (P)")
    print("="*80)
    print("\nP[i,j] = Probability of transitioning from state i to state j\n")

    P = np.array(params['P'])
    df_P = pd.DataFrame(
        P,
        index=[f"From {state_labels[i]}" for i in range(3)],
        columns=[f"To {state_labels[i]}" for i in range(3)]
    )
    print(df_P.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n\nPersistence Probabilities (staying in same state):")
    for i in range(3):
        print(f"  {state_labels[i]}: {P[i,i]:.6f} ({P[i,i]*100:.2f}%)")

    print("\n" + "="*80)
    print("FEATURE MEANS BY STATE")
    print("="*80)

    # Create DataFrame for feature means
    means_data = []
    for state_idx in range(3):
        state_key = f"state_{state_idx}"
        row = {'State': state_labels[state_idx]}
        row.update(params['means'][state_key])
        means_data.append(row)

    df_means = pd.DataFrame(means_data)
    df_means = df_means.set_index('State')

    print("\n" + df_means.to_string(float_format=lambda x: f"{x:.4f}"))

    # Interpretation guide
    print("\n\nInterpretation Guide:")
    print("  ADX (Average Directional Index):")
    print("    - Higher values = Stronger trend")
    print(f"    - Bullish: {params['means']['state_0']['ADX']:.2f}")
    print(f"    - Neutral: {params['means']['state_1']['ADX']:.2f}")
    print(f"    - Bearish: {params['means']['state_2']['ADX']:.2f}")

    print("\n  URSI (Ultimate RSI):")
    print("    - 0-100 scale, 50 = neutral")
    print(f"    - Bullish: {params['means']['state_0']['URSI']:.2f} (strong upward momentum)")
    print(f"    - Neutral: {params['means']['state_1']['URSI']:.2f} (moderate)")
    print(f"    - Bearish: {params['means']['state_2']['URSI']:.2f} (downward momentum)")

    print("\n  BBWP (Bollinger Band Width Percentile):")
    print("    - 0-100 scale, measures volatility")
    print(f"    - Bullish: {params['means']['state_0']['BBWP']:.2f}")
    print(f"    - Neutral: {params['means']['state_1']['BBWP']:.2f}")
    print(f"    - Bearish: {params['means']['state_2']['BBWP']:.2f}")

    print("\n  BB_PCTB (Bollinger Bands %B):")
    print("    - Position within bands, >1 = above upper, <0 = below lower")
    print(f"    - Bullish: {params['means']['state_0']['BB_PCTB']:.4f} (near upper band)")
    print(f"    - Neutral: {params['means']['state_1']['BB_PCTB']:.4f} (middle)")
    print(f"    - Bearish: {params['means']['state_2']['BB_PCTB']:.4f} (near lower band)")

    print("\n" + "="*80)
    print("RETURN STATISTICS BY STATE")
    print("="*80)

    # Return stats
    return_data = []
    for state_idx, state_key in enumerate(['0', '1', '2']):
        row = {
            'State': state_labels[state_idx],
            'Mean Return': params['return_stats'][state_key]['mean_return'],
            'Std Return': params['return_stats'][state_key]['std_return'],
            'Sharpe': params['return_stats'][state_key]['mean_return'] / params['return_stats'][state_key]['std_return'],
            'Count': params['return_stats'][state_key]['count'],
            'Frequency': params['return_stats'][state_key]['count'] / params['n_obs']
        }
        return_data.append(row)

    df_returns = pd.DataFrame(return_data)
    df_returns = df_returns.set_index('State')

    print("\n" + df_returns.to_string(float_format=lambda x: f"{x:.6f}" if abs(x) < 1 else f"{x:.2f}"))

    print("\n\nAnnualized Metrics (assuming 252 trading days):")
    for state_idx, state_key in enumerate(['0', '1', '2']):
        mean_ret = params['return_stats'][state_key]['mean_return']
        std_ret = params['return_stats'][state_key]['std_return']

        annual_return = mean_ret * 252
        annual_vol = std_ret * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        print(f"\n  {state_labels[state_idx]}:")
        print(f"    Annual Return: {annual_return*100:.2f}%")
        print(f"    Annual Volatility: {annual_vol*100:.2f}%")
        print(f"    Sharpe Ratio: {sharpe:.3f}")

    print("\n" + "="*80)
    print("FEATURE STANDARD DEVIATIONS (Scaled)")
    print("="*80)

    # Standard deviations
    stds_data = []
    for state_idx in range(3):
        row = {'State': state_labels[state_idx]}
        for feat_idx, feat_name in enumerate(params['features']):
            row[feat_name] = params['stds_scaled'][state_idx][feat_idx]
        stds_data.append(row)

    df_stds = pd.DataFrame(stds_data)
    df_stds = df_stds.set_index('State')

    print("\n" + df_stds.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n\nNote: These are scaled standard deviations (after standardization)")

    print("\n" + "="*80)
    print("DATA STANDARDIZATION PARAMETERS")
    print("="*80)

    # Scaler parameters
    print("\nOriginal feature means (before scaling):")
    for feat_idx, feat_name in enumerate(params['features']):
        print(f"  {feat_name:12s}: {params['scaler_params']['mean'][feat_idx]:10.4f}")

    print("\nOriginal feature scales (before scaling):")
    for feat_idx, feat_name in enumerate(params['features']):
        print(f"  {feat_name:12s}: {params['scaler_params']['scale'][feat_idx]:10.4f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Calculate some insights
    print("\n1. State Persistence:")
    for i in range(3):
        avg_duration = 1 / (1 - P[i,i]) if P[i,i] < 1 else float('inf')
        print(f"   {state_labels[i]}: Average duration = {avg_duration:.1f} days")

    print("\n2. Most Likely Transitions:")
    for i in range(3):
        for j in range(3):
            if i != j and P[i,j] > 0.01:
                print(f"   {state_labels[i]} → {state_labels[j]}: {P[i,j]*100:.2f}%")

    print("\n3. Risk-Return Profile:")
    for state_idx, state_key in enumerate(['0', '1', '2']):
        mean_ret = params['return_stats'][state_key]['mean_return']
        std_ret = params['return_stats'][state_key]['std_return']
        print(f"   {state_labels[state_idx]}: Return/Risk ratio = {mean_ret/std_ret:.3f}")

    print("\n" + "="*80)

if __name__ == "__main__":
    display_hmm_parameters()
