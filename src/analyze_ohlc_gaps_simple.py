#!/usr/bin/env python3
"""
Analyze OHLC gaps (high-close, low-close) by regime
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_ohlc_gaps_by_regime():
    """Calculate OHLC gap statistics by regime"""

    print("="*80)
    print("OHLC GAP ANALYSIS BY REGIME")
    print("="*80)

    # Load OHLC data
    print("\nLoading OHLC data...")
    df_ohlc = pd.read_csv('data/vnindex_full.csv')
    df_ohlc['date'] = pd.to_datetime(df_ohlc['date'])

    # Rename columns to standard names
    df_ohlc = df_ohlc.rename(columns={
        'OPENINDEX': 'open',
        'HIGHESTINDEX': 'high',
        'LOWESTINDEX': 'low',
        'CLOSEINDEX': 'close'
    })

    print(f"OHLC data shape: {df_ohlc.shape}")

    # Load HMM state assignments
    print("Loading HMM state assignments...")
    df_hmm = pd.read_csv('outputs/step1_params/hmm_states_K3.csv')
    df_hmm['date'] = pd.to_datetime(df_hmm['date'])

    print(f"HMM states data shape: {df_hmm.shape}")

    # Merge
    print("Merging data...")
    df = df_ohlc.merge(df_hmm[['date', 'hmm_state']], on='date', how='inner')
    df = df.rename(columns={'hmm_state': 'regime'})
    print(f"Merged data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Calculate percentage gaps
    df['high_close_gap_pct'] = (df['high'] - df['close']) / df['close'] * 100
    df['low_close_gap_pct'] = (df['close'] - df['low']) / df['close'] * 100

    print("\n" + "="*80)
    print("RESULTS: OHLC GAP STATISTICS BY REGIME")
    print("="*80)

    # Calculate statistics by state
    states = sorted(df['regime'].unique())

    # State labels
    state_labels = {
        0: "Bullish (State 0)",
        1: "Neutral (State 1)",
        2: "Bearish (State 2)"
    }

    results = []

    for state in states:
        state_data = df[df['regime'] == state]

        result = {
            'State': state_labels.get(state, f"State {state}"),
            'Count': len(state_data),
            'High-Close Mean (%)': state_data['high_close_gap_pct'].mean(),
            'High-Close Std (%)': state_data['high_close_gap_pct'].std(),
            'High-Close Min (%)': state_data['high_close_gap_pct'].min(),
            'High-Close Max (%)': state_data['high_close_gap_pct'].max(),
            'Low-Close Mean (%)': state_data['low_close_gap_pct'].mean(),
            'Low-Close Std (%)': state_data['low_close_gap_pct'].std(),
            'Low-Close Min (%)': state_data['low_close_gap_pct'].min(),
            'Low-Close Max (%)': state_data['low_close_gap_pct'].max(),
        }
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('State')

    print("\nSummary Table:")
    print(df_results.to_string(float_format=lambda x: f"{x:.4f}"))

    # Print detailed summary
    print("\n" + "="*80)
    print("DETAILED SUMMARY")
    print("="*80)

    for state in states:
        state_data = df[df['regime'] == state]
        state_label = state_labels.get(state, f"State {state}")

        print(f"\n{state_label}:")
        print(f"  Sample size: {len(state_data)} days")

        print(f"\n  High - Close Gap:")
        print(f"    Mean: {state_data['high_close_gap_pct'].mean():.4f}%")
        print(f"    Std:  {state_data['high_close_gap_pct'].std():.4f}%")
        print(f"    Range: [{state_data['high_close_gap_pct'].min():.4f}%, {state_data['high_close_gap_pct'].max():.4f}%]")
        print(f"    Median: {state_data['high_close_gap_pct'].median():.4f}%")
        print(f"    Q25: {state_data['high_close_gap_pct'].quantile(0.25):.4f}%")
        print(f"    Q75: {state_data['high_close_gap_pct'].quantile(0.75):.4f}%")

        print(f"\n  Close - Low Gap:")
        print(f"    Mean: {state_data['low_close_gap_pct'].mean():.4f}%")
        print(f"    Std:  {state_data['low_close_gap_pct'].std():.4f}%")
        print(f"    Range: [{state_data['low_close_gap_pct'].min():.4f}%, {state_data['low_close_gap_pct'].max():.4f}%]")
        print(f"    Median: {state_data['low_close_gap_pct'].median():.4f}%")
        print(f"    Q25: {state_data['low_close_gap_pct'].quantile(0.25):.4f}%")
        print(f"    Q75: {state_data['low_close_gap_pct'].quantile(0.75):.4f}%")

        # Average total daily range
        daily_range = state_data['high_close_gap_pct'] + state_data['low_close_gap_pct']
        print(f"\n  Total Daily Range (High-Low as % of Close):")
        print(f"    Mean: {daily_range.mean():.4f}%")
        print(f"    Std:  {daily_range.std():.4f}%")
        print(f"    Median: {daily_range.median():.4f}%")

    # Save results
    output_dir = Path('outputs/ohlc_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(output_dir / 'ohlc_gaps_by_regime.csv')

    # Also save the merged data with gaps for further analysis
    df[['date', 'open', 'high', 'low', 'close', 'regime',
        'high_close_gap_pct', 'low_close_gap_pct']].to_csv(
        output_dir / 'daily_ohlc_with_gaps.csv', index=False
    )

    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"  - ohlc_gaps_by_regime.csv")
    print(f"  - daily_ohlc_with_gaps.csv")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Compare across regimes
    print("\nComparison across regimes:")
    print(f"\nHigh-Close Gap (mean %):")
    for state in states:
        state_data = df[df['regime'] == state]
        state_label = state_labels.get(state, f"State {state}")
        mean_gap = state_data['high_close_gap_pct'].mean()
        print(f"  {state_label}: {mean_gap:.4f}%")

    print(f"\nLow-Close Gap (mean %):")
    for state in states:
        state_data = df[df['regime'] == state]
        state_label = state_labels.get(state, f"State {state}")
        mean_gap = state_data['low_close_gap_pct'].mean()
        print(f"  {state_label}: {mean_gap:.4f}%")

    print(f"\nTotal Daily Range (mean %):")
    for state in states:
        state_data = df[df['regime'] == state]
        state_label = state_labels.get(state, f"State {state}")
        daily_range = state_data['high_close_gap_pct'] + state_data['low_close_gap_pct']
        mean_range = daily_range.mean()
        print(f"  {state_label}: {mean_range:.4f}%")

    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_ohlc_gaps_by_regime()
