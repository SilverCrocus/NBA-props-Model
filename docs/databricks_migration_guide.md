# Databricks Migration Guide for NBA Props Model

## Error Diagnosis and Resolution

### Issues Identified

1. **Primary Issue: Categorical Features in Scaling**
   - **Error**: `ValueError: could not convert string to float: 'Guard'`
   - **Cause**: RobustScaler attempted to scale categorical columns (`Position_Inferred`, `Player_Role`)
   - **Solution**: Implemented one-hot encoding before scaling

2. **MLflow Module Error**
   - **Error**: `ModuleNotFoundError: No module named 'modules'`
   - **Cause**: Previous MLflow models saved with custom module dependencies
   - **Solution**: Fresh model training with proper dependency management

### Fixed Implementation

The notebook has been updated with the following fixes:

#### 1. Proper Feature Preprocessing
```python
# Identify categorical and numeric columns
categorical_cols = ['Position_Inferred', 'Player_Role']
numeric_feature_cols = [col for col in df.columns 
                        if col not in exclude_cols + categorical_cols]

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categorical_encoded = encoder.fit_transform(df[categorical_cols])

# Combine numeric and encoded features
X = pd.concat([df[numeric_feature_cols], categorical_df], axis=1)
```

#### 2. Conditional Scaling
```python
# Scale only for models that need it
if model_name in ['Ridge', 'Lasso', 'ElasticNet', 'Huber', 'SVR']:
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
else:
    predictions = model.predict(X)
```

## Databricks Best Practices Implementation

### 1. Data Management
```python
# Replace file loading with Delta Lake
df = spark.read.table("catalog.schema.nba_player_features").toPandas()

# Or use Unity Catalog
df = spark.sql("SELECT * FROM catalog.schema.nba_player_features").toPandas()
```

### 2. MLflow Integration
```python
# Set up experiment tracking
mlflow.set_experiment("/Shared/nba_props_model_experiments")
mlflow.sklearn.autolog()

# Track each model run
with mlflow.start_run(run_name=model_name):
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2': r2})
    mlflow.sklearn.log_model(model, "model")
```

### 3. Model Registry
```python
# Register best model
model_uri = f"runs:/{best_run.info.run_id}/model"
model_version = mlflow.register_model(model_uri, "nba_props_pra_model")

# Transition to production
client.transition_model_version_stage(
    name="nba_props_pra_model",
    version=model_version.version,
    stage="Production"
)
```

### 4. Production Prediction Function
```python
def predict_pra(player_features_df, model_name="nba_props_pra_model"):
    # Load from registry
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    
    # Preprocess features
    X = preprocess_features(player_features_df)
    
    # Get predictions with confidence intervals
    predictions = model.predict(X)
    ci_lower = quantile_models[0.1].predict(X)
    ci_upper = quantile_models[0.9].predict(X)
    
    return pd.DataFrame({
        'prediction': predictions,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
```

## Performance Optimizations for Databricks

### 1. Distributed Processing
```python
# Use Spark for large-scale feature engineering
from pyspark.sql import functions as F

# Calculate rolling averages in Spark
player_stats_df = spark.table("player_game_logs")
rolling_stats = player_stats_df.groupBy("player_id").agg(
    F.avg("PRA").over(Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-5, -1)).alias("PRA_L5"),
    F.avg("PRA").over(Window.partitionBy("player_id").orderBy("game_date").rowsBetween(-10, -1)).alias("PRA_L10")
)
```

### 2. Cluster Configuration
```yaml
# Recommended cluster config for model training
cluster_config:
  spark_version: "13.3.x-ml-scala2.12"
  node_type_id: "Standard_DS3_v2"
  num_workers: 2-8  # Autoscaling
  spark_conf:
    "spark.sql.adaptive.enabled": "true"
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
```

### 3. Job Scheduling
```python
# Daily prediction workflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()

job = w.jobs.create(
    name="nba_props_daily_predictions",
    tasks=[
        jobs.Task(
            task_key="load_features",
            notebook_task=jobs.NotebookTask(
                notebook_path="/Shared/notebooks/load_features"
            )
        ),
        jobs.Task(
            task_key="generate_predictions",
            depends_on=[jobs.TaskDependency(task_key="load_features")],
            notebook_task=jobs.NotebookTask(
                notebook_path="/Shared/notebooks/predict"
            )
        )
    ],
    schedule=jobs.CronSchedule(
        quartz_cron_expression="0 0 10 * * ?",  # Daily at 10 AM
        timezone_id="America/New_York"
    )
)
```

## Files Updated

1. **Original Notebook (Fixed)**: `/notebooks/04_model_evaluation.ipynb`
   - Added categorical encoding
   - Fixed scaling pipeline
   - Updated model saving

2. **Databricks Optimized Version**: `/notebooks/04_model_evaluation_databricks.py`
   - Full MLflow integration
   - Model registry setup
   - Production deployment ready

## Testing Checklist

- [x] Categorical features properly encoded
- [x] Scaling only applied to numeric features
- [x] Models train without errors
- [x] Feature importance handles encoded features
- [x] Model artifacts saved correctly
- [x] Confidence intervals generated
- [ ] MLflow tracking in Databricks
- [ ] Model registry deployment
- [ ] Delta Lake integration
- [ ] Scheduled job setup

## Next Steps

1. **Immediate Actions**
   - Run updated notebook in Databricks
   - Verify MLflow tracking works
   - Test model registry deployment

2. **Data Pipeline**
   - Migrate data to Delta Lake
   - Set up incremental data updates
   - Implement feature store

3. **Production Deployment**
   - Create REST API endpoint
   - Set up monitoring dashboard
   - Implement A/B testing framework

4. **Advanced Features**
   - Add game-by-game features
   - Implement opponent adjustments
   - Include injury/lineup data
   - Add betting line movement tracking

## Troubleshooting

### Common Issues and Solutions

1. **Import Errors**
   ```python
   # Install missing packages
   %pip install xgboost lightgbm scikit-learn
   ```

2. **Memory Issues**
   ```python
   # Use sampling for large datasets
   df_sample = df.sample(frac=0.1, random_state=42)
   ```

3. **Cluster Timeouts**
   ```python
   # Increase cluster size or use caching
   spark.conf.set("spark.sql.shuffle.partitions", "200")
   df.cache()
   ```

## Contact

For questions or issues with the migration, please refer to:
- Databricks documentation: https://docs.databricks.com/
- MLflow documentation: https://mlflow.org/docs/latest/
- Model artifacts: `/models/` directory