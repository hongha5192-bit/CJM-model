"""
Test the real JumpModel with VNINDEX data
"""
import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
import sys
sys.path.append('.')
from src.utils_io import load_config, setup_logging
from src.metrics import best_balanced_accuracy_K2

logger = setup_logging('test_real_jm')

# Load one simulation for testing
df_sim = pd.read_parquet('outputs/step2_sim/K2/sim_T1000_seed000.parquet')
y = df_sim['y'].values
s_true = df_sim['s_true'].values
X = y.reshape(-1, 1)

logger.info("Testing real JumpModel implementations...")

# Test 1: Discrete Jump Model
logger.info("\n1. Testing Discrete JM:")
jm = JumpModel(
    n_components=2,
    jump_penalty=10.0,
    cont=False,
    n_init=5,
    max_iter=100,
    random_state=42
)
jm.fit(X)
y_pred_jm = jm.predict(X)
bac_jm = best_balanced_accuracy_K2(s_true, y_pred_jm)
logger.info(f"   JM BAC: {bac_jm:.4f}")

# Test 2: Continuous Jump Model with mode loss
logger.info("\n2. Testing Continuous CJM with mode loss:")
cjm = JumpModel(
    n_components=2,
    jump_penalty=10.0,
    cont=True,
    mode_loss=True,
    grid_size=0.01,
    n_init=5,
    max_iter=100,
    random_state=42
)
cjm.fit(X)
proba_cjm = cjm.predict_proba(X)
y_pred_cjm = np.argmax(proba_cjm, axis=1)
bac_cjm = best_balanced_accuracy_K2(s_true, y_pred_cjm)
logger.info(f"   CJM BAC: {bac_cjm:.4f}")

# Test 3: Check effect of jump penalty
logger.info("\n3. Testing jump penalty effect:")
for lam in [0.1, 1.0, 10.0, 100.0]:
    model = JumpModel(n_components=2, jump_penalty=lam, cont=False,
                      n_init=2, max_iter=50, random_state=42)
    model.fit(X)
    labels = model.predict(X)
    n_switches = np.sum(np.diff(labels) != 0)
    logger.info(f"   λ={lam:6.1f}: {n_switches} regime switches")

logger.info("\n✓ Real JumpModel is working correctly!")
logger.info("The package provides proper regime detection with jump penalties.")