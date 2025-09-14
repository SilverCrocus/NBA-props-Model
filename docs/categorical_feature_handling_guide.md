# Categorical Feature Handling Guide for NBA Props Model

## Overview
This guide explains how to properly handle categorical features (`Position_Inferred` and `Player_Role`) for different model types in your NBA props prediction model.

## The Problem
Your categorical features have string values:
- **Position_Inferred**: 'Guard', 'Forward', 'Big'
- **Player_Role**: 'Primary', 'Secondary', 'Role', 'Bench'

These cannot be directly used with numerical models and scalers like RobustScaler.

## Solution: Model-Specific Encoding

### 1. Tree-Based Models (XGBoost, RandomForest, LightGBM, ExtraTrees)

**Best Encoding**: Ordinal Encoding or Label Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

# For tree models
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_encoded = ordinal_encoder.fit_transform(df[categorical_cols])
```

**Why this works:**
- Trees can naturally handle the ordinal nature of encoded features
- More memory efficient (1 feature instead of N for N categories)
- Faster training and prediction
- Trees will find optimal splits regardless of the encoding order

**Example mapping:**
- Position_Inferred: Guard→0, Forward→1, Big→2
- Player_Role: Primary→0, Secondary→1, Role→2, Bench→3

### 2. Linear Models (Ridge, Lasso, ElasticNet, SVR, Huber)

**Best Encoding**: One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

# For linear models
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
X_encoded = onehot_encoder.fit_transform(df[categorical_cols])
```

**Why this is necessary:**
- Linear models assume numerical relationships between feature values
- Ordinal encoding would incorrectly imply Guard < Forward < Big
- One-hot encoding treats each category independently
- Each category gets its own coefficient

**Example transformation:**
- Position_Inferred → Position_Forward, Position_Big (Guard dropped as reference)
- Player_Role → Player_Role_Secondary, Player_Role_Role, Player_Role_Bench (Primary dropped)

## Implementation Strategy

### Using ColumnTransformer and Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder

def create_preprocessor(model_type='tree'):
    if model_type == 'tree':
        # Tree-based models
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),  # No scaling needed
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', 
                                       unknown_value=-1), categorical_cols)
            ]
        )
    else:
        # Linear models
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_cols),  # Scale numeric features
                ('cat', OneHotEncoder(sparse_output=False, 
                                      handle_unknown='ignore', 
                                      drop='first'), categorical_cols)
            ]
        )
    return preprocessor
```

## Feature Importance Interpretation

### For Tree-Based Models
- Direct importance value for each categorical feature
- Example: `Position_Inferred: 0.0234` means position explains 2.34% of splits

### For Linear Models with One-Hot Encoding
- Each category has its own importance/coefficient
- Aggregate by summing across categories:

```python
def aggregate_categorical_importance(feature_importances, feature_names):
    aggregated = {}
    for feat, imp in zip(feature_names, feature_importances):
        if 'Position_' in feat or 'Player_Role_' in feat:
            # Extract base feature name
            base = feat.split('_')[0] + '_' + feat.split('_')[1]
            if base not in aggregated:
                aggregated[base] = 0
            aggregated[base] += abs(imp)
        else:
            aggregated[feat] = imp
    return aggregated
```

## Best Practices

### 1. Handle Unknown Categories
Always use `handle_unknown` parameter:
- **Ordinal**: `handle_unknown='use_encoded_value', unknown_value=-1`
- **One-Hot**: `handle_unknown='ignore'`

### 2. Avoid Multicollinearity in Linear Models
Use `drop='first'` in OneHotEncoder to drop one category as reference

### 3. Consistent Train/Test Processing
```python
# Fit on training data only
encoder.fit(X_train[categorical_cols])

# Transform both train and test
X_train_encoded = encoder.transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])
```

### 4. Save Encoders for Production
```python
import joblib

# Save fitted encoders
joblib.dump(encoder, 'models/encoder.pkl')

# Load for predictions
encoder = joblib.load('models/encoder.pkl')
X_new_encoded = encoder.transform(X_new[categorical_cols])
```

## Model Performance Impact

### Expected Performance Differences

| Model Type | Without Proper Encoding | With Proper Encoding | Improvement |
|------------|------------------------|---------------------|-------------|
| XGBoost | Error/Poor | Good | Significant |
| RandomForest | Error/Poor | Good | Significant |
| Ridge/Lasso | Error | Good | Enables training |
| SVR | Error | Good | Enables training |

### Typical Feature Importance Rankings

For NBA player props, expect:
1. **High Importance**: Minutes played, recent performance metrics
2. **Medium Importance**: Team metrics, opponent strength
3. **Low-Medium Importance**: Position_Inferred, Player_Role
4. **Context Dependent**: Injury status, rest days

## Common Pitfalls to Avoid

### ❌ Don't Do This:
```python
# Wrong: Scaling categorical features as strings
scaler.fit_transform(df)  # Will fail with string categories

# Wrong: Manual integer encoding without tracking
df['Position_Inferred'] = df['Position_Inferred'].map({'Guard': 0, 'Forward': 1, 'Big': 2})
# Problem: No handling of unknown values, manual maintenance
```

### ✅ Do This Instead:
```python
# Right: Use proper encoders with pipelines
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)
X_processed = preprocessor.fit_transform(X)
```

## Production Deployment Considerations

### 1. Feature Consistency
Ensure the same features are available in production:
```python
required_features = {
    'numeric': numeric_cols,
    'categorical': categorical_cols
}
```

### 2. Handling New Categories
New player roles or positions may appear:
- One-hot: Automatically ignored (zero vector)
- Ordinal: Mapped to -1 (unknown value)

### 3. Model Monitoring
Track distribution shifts in categorical features:
```python
def check_category_distribution(train_dist, prod_dist):
    for cat in categorical_cols:
        train_counts = train_dist[cat].value_counts(normalize=True)
        prod_counts = prod_dist[cat].value_counts(normalize=True)
        
        # Alert if significant distribution shift
        if abs(train_counts - prod_counts).max() > 0.1:
            print(f"Warning: Distribution shift in {cat}")
```

## Conclusion

Proper categorical feature handling is crucial for model performance. Use:
- **Ordinal encoding** for tree-based models (memory efficient, natural handling)
- **One-hot encoding** for linear models (avoids false ordinal relationships)
- **ColumnTransformer** for clean, reproducible pipelines
- **Proper handling** of unknown categories in production

This approach ensures optimal model performance and robust production deployment.