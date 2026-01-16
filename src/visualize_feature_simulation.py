#!/usr/bin/env python
"""
Visualize and compare feature-based HMM simulation with real data
"""
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_simulation_sample():
    """Load a sample simulation from feature-based HMM"""
    sim_dir = Path('outputs/step2_sim/K3_features')

    # Load first simulation as sample
    sim_file = sim_dir / 'sim_T1000_00000.parquet'
    if not sim_file.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_file}")

    df_sim = pd.read_parquet(sim_file)
    df_sim['date'] = pd.date_range('2018-01-01', periods=len(df_sim), freq='D')

    return df_sim


def load_real_data():
    """Load real VNINDEX data with features"""
    df_real = pd.read_csv('outputs/prepared_data.csv', parse_dates=['date'])

    # Load HMM predictions for real data
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    # For simplicity, we'll use a basic state assignment based on URSI levels
    # (In practice, you'd use Viterbi algorithm)
    df_real['state_approx'] = pd.cut(
        df_real['URSI'],
        bins=[-np.inf, 50, 75, np.inf],
        labels=[2, 1, 0]  # Bearish, Neutral, Bullish
    ).astype(int)

    return df_real


def create_feature_comparison_plot(df_sim, df_real):
    """Create comparison plots for simulated vs real features"""

    # Create subplots for each feature
    features = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'ADX (Trend Strength)', 'URSI (Momentum)',
            'BBWP (Volatility Percentile)', 'BB %B (Price Position)',
            'URSI 0-50 (Market Breadth)', 'URSI 0-20 (Extreme Breadth)'
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12
    )

    # Define colors for states
    state_colors = {0: '#00cc00', 1: '#808080', 2: '#ff3333'}

    # Plot each feature
    for idx, feat in enumerate(features):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Add simulated data trace
        fig.add_trace(
            go.Scatter(
                x=df_sim['date'],
                y=df_sim[feat],
                mode='lines',
                name=f'{feat} (Sim)',
                line=dict(color='blue', width=1),
                opacity=0.7,
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Add real data trace (sample last 1000 days)
        df_real_sample = df_real.tail(1000)
        fig.add_trace(
            go.Scatter(
                x=df_real_sample['date'],
                y=df_real_sample[feat],
                mode='lines',
                name=f'{feat} (Real)',
                line=dict(color='red', width=1),
                opacity=0.7,
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )

        # Add state coloring for simulation
        for state in [0, 1, 2]:
            state_mask = df_sim['s_true'] == state
            if state_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=df_sim.loc[state_mask, 'date'],
                        y=df_sim.loc[state_mask, feat],
                        mode='markers',
                        name=f'State {state}',
                        marker=dict(
                            color=state_colors[state],
                            size=3,
                            opacity=0.3
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )

    # Update layout
    fig.update_layout(
        title='Feature-Based HMM Simulation vs Real Data Comparison',
        height=1200,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis labels
    for idx, feat in enumerate(features):
        row = idx // 2 + 1
        col = idx % 2 + 1
        fig.update_yaxes(title_text=feat, row=row, col=col)

    return fig


def create_state_distribution_comparison(df_sim, df_real):
    """Compare state distributions between simulation and real data"""

    # Calculate state frequencies for simulation
    sim_state_freq = df_sim['s_true'].value_counts(normalize=True).sort_index()

    # Load HMM parameters for expected frequencies
    params_path = Path('outputs/step1_params/hmm_K3_features.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    expected_freq = params['pi']

    # Create bar chart
    fig = go.Figure()

    # Expected frequencies (from stationary distribution)
    fig.add_trace(go.Bar(
        x=['State 0 (Bullish)', 'State 1 (Neutral)', 'State 2 (Bearish)'],
        y=expected_freq,
        name='Expected (π)',
        marker_color='lightblue',
        text=[f'{v:.1%}' for v in expected_freq],
        textposition='auto'
    ))

    # Simulated frequencies
    fig.add_trace(go.Bar(
        x=['State 0 (Bullish)', 'State 1 (Neutral)', 'State 2 (Bearish)'],
        y=[sim_state_freq.get(i, 0) for i in range(3)],
        name='Simulated',
        marker_color='darkblue',
        text=[f'{sim_state_freq.get(i, 0):.1%}' for i in range(3)],
        textposition='auto'
    ))

    fig.update_layout(
        title='State Distribution: Expected vs Simulated',
        xaxis_title='State',
        yaxis_title='Frequency',
        yaxis=dict(tickformat='.0%'),
        barmode='group',
        height=400,
        width=800
    )

    return fig


def create_return_distribution_plot(df_sim, df_real):
    """Compare return distributions by state"""

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            'State 0: Bullish',
            'State 1: Neutral',
            'State 2: Bearish'
        ],
        horizontal_spacing=0.12
    )

    # Colors for each state
    colors = ['#00cc00', '#808080', '#ff3333']

    for state in range(3):
        # Simulated returns for this state
        sim_returns = df_sim[df_sim['s_true'] == state]['y']

        # Add histogram for simulated returns
        fig.add_trace(
            go.Histogram(
                x=sim_returns,
                name=f'Simulated',
                marker_color=colors[state],
                opacity=0.6,
                nbinsx=30,
                histnorm='probability density',
                showlegend=(state == 0)
            ),
            row=1, col=state+1
        )

        # Add normal distribution overlay
        mean = sim_returns.mean()
        std = sim_returns.std()
        x_range = np.linspace(mean - 4*std, mean + 4*std, 100)
        y_normal = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std)**2)

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Normal fit',
                line=dict(color='black', width=2),
                showlegend=(state == 0)
            ),
            row=1, col=state+1
        )

        # Add vertical line for mean
        fig.add_vline(
            x=mean,
            line_dash="dash",
            line_color="red",
            annotation_text=f"μ={mean:.3f}",
            row=1, col=state+1
        )

    fig.update_layout(
        title='Return Distributions by State (Feature-Based HMM Simulation)',
        height=400,
        width=1200,
        showlegend=True
    )

    # Update x-axes
    for col in range(1, 4):
        fig.update_xaxes(title_text='Return', row=1, col=col)

    fig.update_yaxes(title_text='Density', row=1, col=1)

    return fig


def main():
    """Main function to create all visualizations"""

    logger.info("Loading data...")

    # Load simulation and real data
    df_sim = load_simulation_sample()
    df_real = load_real_data()

    logger.info(f"Loaded simulation with {len(df_sim)} days")
    logger.info(f"Loaded real data with {len(df_real)} days")

    # Create output directory
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create feature comparison plot
    logger.info("Creating feature comparison plot...")
    fig_features = create_feature_comparison_plot(df_sim, df_real)
    fig_features.write_html(output_dir / 'feature_simulation_comparison.html')

    # Create state distribution comparison
    logger.info("Creating state distribution comparison...")
    fig_states = create_state_distribution_comparison(df_sim, df_real)
    fig_states.write_html(output_dir / 'state_distribution_comparison.html')

    # Create return distribution plot
    logger.info("Creating return distribution plot...")
    fig_returns = create_return_distribution_plot(df_sim, df_real)
    fig_returns.write_html(output_dir / 'return_distributions_by_state.html')

    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("FEATURE-BASED HMM SIMULATION SUMMARY")
    logger.info("="*80)

    # State statistics
    state_counts = df_sim['s_true'].value_counts().sort_index()
    logger.info("\nState Distribution in Sample Simulation:")
    for state in range(3):
        count = state_counts.get(state, 0)
        pct = count / len(df_sim) * 100
        logger.info(f"  State {state}: {count:4d} days ({pct:5.1f}%)")

    # Feature means by state
    logger.info("\nFeature Means by State (Simulation):")
    features = ['ADX', 'URSI', 'BBWP', 'BB_PCTB', 'URSI_0_50', 'URSI_0_20']
    for state in range(3):
        state_data = df_sim[df_sim['s_true'] == state]
        logger.info(f"\n  State {state}:")
        for feat in features:
            mean_val = state_data[feat].mean()
            logger.info(f"    {feat:10s}: {mean_val:7.2f}")

    # Return statistics by state
    logger.info("\nReturn Statistics by State (Simulation):")
    for state in range(3):
        state_returns = df_sim[df_sim['s_true'] == state]['y']
        logger.info(f"  State {state}: μ={state_returns.mean():7.4f}, σ={state_returns.std():7.4f}")

    logger.info("\n✅ Visualizations saved to outputs/visualizations/")
    logger.info("  - feature_simulation_comparison.html")
    logger.info("  - state_distribution_comparison.html")
    logger.info("  - return_distributions_by_state.html")


if __name__ == '__main__':
    main()