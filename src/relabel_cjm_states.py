#!/usr/bin/env python3
"""
Re-label CJM states by return characteristics to make them comparable with HMM
"""
import pandas as pd
import numpy as np

def relabel_cjm_states():
    """Re-label CJM states to match interpretation: Bullish/Neutral/Bearish"""

    print("="*80)
    print("RE-LABELING CJM STATES BY RETURN CHARACTERISTICS")
    print("="*80)

    # Load CJM data
    df_cjm = pd.read_csv('outputs/step4_apply/regimes_daily_K3.csv')

    # Calculate return statistics for each state
    state_returns = {}
    for state in range(3):
        state_data = df_cjm[df_cjm['regime'] == state]
        state_returns[state] = {
            'mean_return': state_data['ret'].mean(),
            'std_return': state_data['ret'].std(),
            'count': len(state_data)
        }

    print("\nOriginal CJM State Statistics:")
    for state in range(3):
        print(f"\nState {state}:")
        print(f"  Mean return: {state_returns[state]['mean_return']*100:.4f}%")
        print(f"  Std return:  {state_returns[state]['std_return']*100:.4f}%")
        print(f"  Count: {state_returns[state]['count']} days")

    # Sort states by mean return (high to low)
    sorted_by_return = sorted(state_returns.items(), key=lambda x: x[1]['mean_return'], reverse=True)

    print("\n" + "="*80)
    print("RELABELING SCHEME")
    print("="*80)
    print("\nSorted by mean return (high to low):")

    # Create mapping
    # Highest return -> Bullish (0)
    # Middle return -> Neutral (1)
    # Lowest return -> Bearish (2)

    old_to_new = {}
    labels = ['Bullish (0)', 'Neutral (1)', 'Bearish (2)']

    for new_state, (old_state, stats) in enumerate(sorted_by_return):
        old_to_new[old_state] = new_state
        print(f"\nOld State {old_state} → New State {new_state} ({labels[new_state]})")
        print(f"  Mean return: {stats['mean_return']*100:.4f}%")
        print(f"  Std return:  {stats['std_return']*100:.4f}%")

    # Apply relabeling
    df_cjm['regime_relabeled'] = df_cjm['regime'].map(old_to_new)

    # Verify
    print("\n" + "="*80)
    print("VERIFICATION - RELABELED CJM STATES")
    print("="*80)

    for new_state in range(3):
        state_data = df_cjm[df_cjm['regime_relabeled'] == new_state]
        print(f"\n{labels[new_state]}:")
        print(f"  Count: {len(state_data)} days ({len(state_data)/len(df_cjm)*100:.1f}%)")
        print(f"  Mean return: {state_data['ret'].mean()*100:.4f}%")
        print(f"  Std return:  {state_data['ret'].std()*100:.4f}%")

    # Save
    output_path = 'outputs/step4_apply/regimes_daily_K3_relabeled.csv'
    df_cjm.to_csv(output_path, index=False)

    print(f"\n✓ Saved relabeled CJM states to: {output_path}")
    print("\n" + "="*80)

    return old_to_new

if __name__ == "__main__":
    relabel_cjm_states()
