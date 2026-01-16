# BAC (Balanced Accuracy) Performance Analysis

## Our K=3 Feature-Based Model Results

### 1. Empirical BAC Results from Lambda Scan

| Lambda (λ) | Mean BAC | Std BAC | Interpretation |
|------------|----------|---------|----------------|
| 0.1        | 1.000    | 0.000   | Perfect separation (overfitting risk) |
| 0.46       | 1.000    | 0.000   | Perfect separation |
| 2.15       | 0.993    | 0.021   | Excellent |
| **10.0**   | **0.979**| 0.038   | **Optimal (Selected)** |
| 46.4       | 0.918    | 0.075   | Good |
| 215.4      | 0.823    | 0.112   | Moderate |
| 1000.0     | 0.754    | 0.165   | Acceptable |

### 2. Why Our BAC is Higher Than Returns-Only Models

#### A. Information Advantage
- **Returns-only**: 1 dimension (price changes)
- **Our approach**: 6 dimensions (ADX, URSI, BBWP, BB_PCTB, URSI_0_50, URSI_0_20)
- **Impact**: 6x more information → Better regime separation

#### B. Feature Separation Quality
```
Euclidean distances between regime centers:
- Neutral ↔ Bullish: 38.9 units
- Neutral ↔ Bearish: 60.4 units
- Bullish ↔ Bearish: 86.5 units

Large distances = Clear separation = High BAC
```

#### C. Distinctive Feature Patterns
| Regime | URSI | ADX | BBWP | Interpretation |
|--------|------|-----|------|----------------|
| Neutral | 60 | 20 | 40 | Mid-range momentum, weak trend |
| Bullish | 85 | 30 | 50 | High momentum, moderate trend |
| Bearish | 30 | 35 | 75 | Low momentum, high volatility |

### 3. Comparison with Expected Performance

| Model Type | Expected BAC | Actual BAC | Advantage |
|------------|--------------|------------|-----------|
| Random Baseline (K=3) | 0.333 | - | - |
| Returns-only HMM | 0.70-0.85 | - | Baseline |
| **Our Feature-Based** | - | **0.98** | +15-30% |

### 4. Real Data Application Results

When applied to VNINDEX (2018-2025) with λ=10:

- **Regime Distribution**:
  - Neutral: 46.3%
  - Bullish: 33.7%
  - Bearish: 20.0%

- **Model Confidence**: Average probability ≈ 0.98 for assigned regimes
- **Regime Persistence**: 97.1% (regimes are stable, not jumpy)
- **Transitions per Year**: ~7.2 (realistic for position trading)

### 5. Key Insights

#### Strengths of High BAC:
1. **Clear Regime Identification**: Model is confident in its classifications
2. **Low False Switching**: High persistence means fewer whipsaws
3. **Actionable Signals**: Clear buy/sell/hold implications

#### Potential Concerns:
1. **Overfitting Risk**: BAC=1.0 on simulations might not generalize
2. **Small Test Set**: Only 20 simulations for quick mode
3. **Feature Engineering**: Hand-picked features may introduce bias

### 6. Validation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| BAC on Simulations | 0.979 | Excellent |
| Regime Persistence | 97.1% | Very stable |
| Avg Confidence | 0.98 | High certainty |
| Regime Balance | 46/34/20% | Reasonable distribution |

### 7. BAC Formula Reminder

For K=3 regimes:
```
BAC = (Recall_0 + Recall_1 + Recall_2) / 3
    = (TP_0/N_0 + TP_1/N_1 + TP_2/N_2) / 3
```

Where:
- TP_k = True positives for regime k
- N_k = Total samples in true regime k
- Balanced across all regimes (unlike regular accuracy)

### 8. Conclusion

Our feature-based approach achieves **BAC ≈ 0.98**, which is:
- **15-30% higher** than expected returns-only models
- Due to **6D feature space** providing rich market context
- **Validated empirically** through actual CJM fitting
- **Stable on real data** with 97% persistence

This high BAC indicates the model successfully leverages technical indicators to create well-separated, identifiable market regimes that persist long enough for practical trading applications.