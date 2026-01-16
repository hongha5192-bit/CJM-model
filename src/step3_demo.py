
import sys
sys.path.append('.')
from src.utils_io import load_config, load_parquet, save_csv, save_json, setup_logging, numpy_to_python
from src.metrics import best_balanced_accuracy_K2
from src.jump_model_mock import JumpModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = setup_logging('step3_demo')

# Load demo config
config = load_config('configs/daily_demo.yaml')

# Create lambda grid
lambdas = np.logspace(-1, 3, 5)  # 0.1, 1, 10, 100, 1000
logger.info(f"Testing lambdas: {lambdas}")

# Load 10 simulations
sim_data = []
for i in range(10):
    df = load_parquet(f"outputs/step2_sim/K2/sim_T0100_seed{i:03d}.parquet")
    sim_data.append((df['y'].values, df['s_true'].values))

logger.info(f"Loaded {len(sim_data)} simulations")

# Test each lambda
results = []
for lam in lambdas:
    logger.info(f"Testing lambda={lam:.1f}")
    for model_type in ['JM', 'CJM_mode']:
        bac_scores = []
        for y, s_true in sim_data:
            try:
                X = y.reshape(-1, 1)
                model = JumpModel(n_components=2, jump_penalty=lam,
                                cont=(model_type=='CJM_mode'),
                                mode_loss=True, n_init=2, max_iter=100,
                                tol=1e-4, random_state=42)
                model.fit(X, ret_ser=y, sort_by="cumret")

                if model_type == 'JM':
                    y_pred = model.predict(X)
                else:
                    proba = model.predict_proba(X)
                    y_pred = proba.values.argmax(axis=1)

                bac = best_balanced_accuracy_K2(s_true, y_pred)
                bac_scores.append(bac)
            except:
                pass

        if bac_scores:
            results.append({
                'K': 2,
                'model_type': model_type,
                'lambda': lam,
                'mean_BAC': np.mean(bac_scores),
                'std_BAC': np.std(bac_scores),
                'n_sim': len(bac_scores)
            })

# Save results
df_results = pd.DataFrame(results)
save_csv(df_results, 'outputs/step3_lambda/lambda_scores.csv')

# Find best lambda
best_lambda = {}
for mt in ['JM', 'CJM_mode']:
    df_mt = df_results[df_results['model_type'] == mt]
    if len(df_mt) > 0:
        best_idx = df_mt['mean_BAC'].idxmax()
        best_lambda[mt] = float(df_mt.loc[best_idx, 'lambda'])

save_json({'K2': best_lambda}, 'outputs/step3_lambda/best_lambda.json')

# Create plot
plt.figure(figsize=(8, 5))
for mt in ['JM', 'CJM_mode']:
    df_mt = df_results[df_results['model_type'] == mt]
    if len(df_mt) > 0:
        plt.plot(df_mt['lambda'], df_mt['mean_BAC'], 'o-', label=mt)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Mean BAC')
plt.title('BAC vs Lambda (Demo)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/step3_lambda/bac_vs_lambda.png', dpi=100)
plt.close()

logger.info("Step 3 demo complete!")
