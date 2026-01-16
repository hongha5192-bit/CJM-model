# HMM Model Comparison: Returns-Only vs Features-Based

## 📊 Key Differences Summary

| Aspect | Returns-Only HMM | Features-Based HMM | Difference |
|--------|------------------|-------------------|------------|
| **Input Data** | 1D (returns only) | 6D (ADX, URSI, BBWP, BB_PCTB, URSI_0_50, URSI_0_20) | 6x more information |
| **Log-Likelihood** | 6,214 | -8,108 | Different scale due to dimensionality |
| **Interpretation** | Volatility-based | Market condition-based | More actionable |

## 🎯 State Parameters Comparison

### Returns-Only HMM (Volatility-Based)

| State | Mean Return | Volatility | Stationary π | Interpretation |
|-------|-------------|------------|--------------|----------------|
| **0** | +0.174% | 0.77% | 77.40% | Low volatility |
| **1** | -0.286% | 2.08% | 7.80% | Medium volatility |
| **2** | -0.581% | 2.39% | 14.81% | High volatility |

### Features-Based HMM (Market Condition-Based)

| State | Mean Return | Volatility | Stationary π | Interpretation |
|-------|-------------|------------|--------------|----------------|
| **0** | +0.202% | 0.92% | 24.91% | Bullish trend |
| **1** | +0.031% | 1.12% | 43.24% | Neutral/Consolidation |
| **2** | -0.131% | 1.59% | 31.85% | Bearish/Stressed |

## 🔄 Transition Matrix Comparison

### Returns-Only Model
```
        To→ State 0    State 1    State 2
From↓
State 0     92.94%      6.05%      1.01%
State 1     69.92%      4.81%     25.27%
State 2      0.07%     18.53%     81.41%
```

### Features-Based Model
```
        To→ State 0    State 1    State 2
From↓
State 0     96.58%      3.42%      0.00%
State 1      1.97%     95.55%      2.48%
State 2      0.00%      3.37%     96.63%
```

## 📈 Key Differences Analysis

### 1. **State Distribution**
- **Returns-Only**: Heavily skewed to State 0 (77.4%)
- **Features-Based**: More balanced (25%, 43%, 32%)
- **Implication**: Features identify distinct market conditions beyond just volatility

### 2. **State Persistence**
- **Returns-Only**: State 1 very unstable (4.81% self-transition)
- **Features-Based**: All states highly persistent (95-97% self-transition)
- **Implication**: Features create more stable, meaningful regimes

### 3. **Transition Patterns**
- **Returns-Only**:
  - State 1 → State 0: 69.92% (quick reversion to low vol)
  - Direct State 0 ↔ State 2 transitions rare
- **Features-Based**:
  - Almost no direct State 0 ↔ State 2 transitions
  - State 1 acts as true transitional state
  - More symmetric transition structure

### 4. **Mean Returns**
- **Returns-Only**: Negative returns in States 1 & 2
- **Features-Based**: Only State 2 has negative returns
- **Implication**: Features better separate profitable from unprofitable regimes

### 5. **Volatility Levels**
| Model | State 0 Vol | State 1 Vol | State 2 Vol | Ratio (High/Low) |
|-------|------------|-------------|-------------|------------------|
| Returns-Only | 0.77% | 2.08% | 2.39% | 3.10x |
| Features-Based | 0.92% | 1.12% | 1.59% | 1.73x |

- Returns-only model shows more extreme volatility separation
- Features-based model shows moderate, more realistic volatility differences

## 🎨 Feature Characteristics (Features-Based Only)

| State | URSI | ADX | BBWP | Market Condition |
|-------|------|-----|------|------------------|
| **0** | 88.4 | 29.9 | 46.5 | Overbought, moderate trend |
| **1** | 67.3 | 20.2 | 41.7 | Neutral momentum, weak trend |
| **2** | 30.5 | 33.1 | 61.3 | Oversold, strong (down) trend |

## 💡 Practical Implications

### Returns-Only Model
✅ **Pros:**
- Simple, focuses on volatility clustering
- Clear volatility regimes
- High persistence in extreme states

❌ **Cons:**
- Limited information
- Unstable middle state
- Less actionable for trading

### Features-Based Model
✅ **Pros:**
- Rich information from 6 features
- Stable, persistent regimes
- Clear market condition identification
- Better trading signals (URSI levels)

❌ **Cons:**
- More complex
- Requires all 6 features for prediction
- Less extreme volatility separation

## 📊 Conclusion

The **features-based HMM** provides:
1. **More balanced regime distribution** (not dominated by one state)
2. **Higher regime persistence** (95-97% vs 5-93%)
3. **Clearer economic interpretation** (Bull/Neutral/Bear vs Low/Med/High vol)
4. **Better separation of profitable regimes** (State 0 & 1 positive returns)
5. **Actionable trading signals** from feature levels

**Recommendation**: Use the features-based model for practical applications as it provides richer, more stable, and more interpretable market regimes.