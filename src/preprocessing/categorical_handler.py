"""
Categorical Feature Handler for NBA Props Model

This module provides utilities for properly handling categorical features
in machine learning pipelines, supporting both tree-based and linear models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import List, Dict, Tuple, Optional, Union


class CategoricalFeatureHandler:
    """
    Handles categorical feature encoding for different model types.
    
    Supports:
    - One-hot encoding for linear models
    - Label/Ordinal encoding for tree-based models
    - Mixed pipelines with proper preprocessing
    """
    
    def __init__(self, 
                 categorical_columns: List[str],
                 numeric_columns: List[str],
                 encoding_strategy: str = 'auto'):
        """
        Initialize the categorical feature handler.
        
        Parameters:
        -----------
        categorical_columns : List[str]
            List of categorical column names
        numeric_columns : List[str]
            List of numeric column names
        encoding_strategy : str
            'onehot', 'label', 'ordinal', or 'auto'
        """
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.encoding_strategy = encoding_strategy
        self.encoders = {}
        self.feature_names = None
        
    def create_preprocessor(self, model_type: str = 'tree') -> ColumnTransformer:
        """
        Create a preprocessor pipeline based on model type.
        
        Parameters:
        -----------
        model_type : str
            'tree' for tree-based models (XGBoost, RandomForest, etc.)
            'linear' for linear models (Ridge, Lasso, etc.)
            
        Returns:
        --------
        ColumnTransformer with appropriate preprocessing
        """
        if model_type == 'tree':
            # Tree-based models can handle ordinal encoding
            categorical_transformer = Pipeline([
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', 
                                         unknown_value=-1))
            ])
            numeric_transformer = 'passthrough'  # Trees don't need scaling
            
        elif model_type == 'linear':
            # Linear models need one-hot encoding and scaling
            categorical_transformer = Pipeline([
                ('encoder', OneHotEncoder(sparse_output=False, 
                                         handle_unknown='ignore',
                                         drop='first'))  # Drop first to avoid multicollinearity
            ])
            numeric_transformer = Pipeline([
                ('scaler', RobustScaler())
            ])
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ],
            remainder='passthrough'  # Keep any other columns
        )
        
        return preprocessor
    
    def fit_transform(self, 
                      X: pd.DataFrame, 
                      model_type: str = 'tree') -> np.ndarray:
        """
        Fit and transform the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        model_type : str
            Type of model ('tree' or 'linear')
            
        Returns:
        --------
        Transformed feature array
        """
        preprocessor = self.create_preprocessor(model_type)
        X_transformed = preprocessor.fit_transform(X)
        
        # Store the preprocessor for later use
        self.preprocessor = preprocessor
        
        # Get feature names after transformation
        self.feature_names = self.get_feature_names(preprocessor, model_type)
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        Transformed feature array
        """
        if not hasattr(self, 'preprocessor'):
            raise ValueError("Must call fit_transform first")
        
        return self.preprocessor.transform(X)
    
    def get_feature_names(self, 
                         preprocessor: ColumnTransformer,
                         model_type: str) -> List[str]:
        """
        Get feature names after transformation.
        
        Parameters:
        -----------
        preprocessor : ColumnTransformer
            Fitted preprocessor
        model_type : str
            Type of model
            
        Returns:
        --------
        List of feature names
        """
        feature_names = []
        
        # Numeric features (unchanged names)
        feature_names.extend(self.numeric_columns)
        
        # Categorical features
        if model_type == 'tree':
            # Ordinal encoding keeps original names
            feature_names.extend(self.categorical_columns)
        else:
            # One-hot encoding creates new names
            cat_transformer = preprocessor.named_transformers_['cat']
            encoder = cat_transformer.named_steps['encoder']
            
            # Get one-hot encoded feature names
            for i, col in enumerate(self.categorical_columns):
                categories = encoder.categories_[i]
                # Skip first category if drop='first'
                if encoder.drop is not None:
                    categories = categories[1:]
                for category in categories:
                    feature_names.append(f"{col}_{category}")
        
        return feature_names
    
    def encode_for_multiple_models(self, 
                                   X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create multiple encoded versions for different model types.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        Dictionary with encoded dataframes for each model type
        """
        encoded_data = {}
        
        # For tree-based models
        tree_preprocessor = self.create_preprocessor('tree')
        X_tree = tree_preprocessor.fit_transform(X)
        tree_features = self.get_feature_names(tree_preprocessor, 'tree')
        encoded_data['tree'] = pd.DataFrame(X_tree, columns=tree_features, index=X.index)
        
        # For linear models
        linear_preprocessor = self.create_preprocessor('linear')
        X_linear = linear_preprocessor.fit_transform(X)
        linear_features = self.get_feature_names(linear_preprocessor, 'linear')
        encoded_data['linear'] = pd.DataFrame(X_linear, columns=linear_features, index=X.index)
        
        # Store preprocessors
        self.preprocessors = {
            'tree': tree_preprocessor,
            'linear': linear_preprocessor
        }
        
        return encoded_data


def prepare_features_for_modeling(df: pd.DataFrame,
                                  target_col: str = 'PRA_estimate',
                                  categorical_cols: Optional[List[str]] = None,
                                  exclude_cols: Optional[List[str]] = None) -> Tuple[Dict, pd.Series, CategoricalFeatureHandler]:
    """
    Prepare features for multiple model types with proper categorical handling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    target_col : str
        Name of target column
    categorical_cols : List[str]
        List of categorical columns (auto-detected if None)
    exclude_cols : List[str]
        Columns to exclude from features
        
    Returns:
    --------
    Tuple of (encoded_features_dict, target, handler)
    """
    if categorical_cols is None:
        # Auto-detect categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Add target to exclude list
    exclude_cols.append(target_col)
    
    # Get numeric columns
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    
    # Remove categorical and excluded columns from numeric
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]
    
    # Get feature columns
    feature_df = df[numeric_cols + categorical_cols].copy()
    
    # Create handler
    handler = CategoricalFeatureHandler(
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols
    )
    
    # Encode for different model types
    encoded_features = handler.encode_for_multiple_models(feature_df)
    
    # Get target
    y = df[target_col].copy()
    
    return encoded_features, y, handler


def get_feature_importance_with_categories(model,
                                          feature_names: List[str],
                                          categorical_mappings: Optional[Dict] = None) -> pd.DataFrame:
    """
    Get feature importance and aggregate one-hot encoded features.
    
    Parameters:
    -----------
    model : sklearn model
        Fitted model with feature_importances_
    feature_names : List[str]
        List of feature names
    categorical_mappings : Dict
        Mapping of original categorical columns to encoded columns
        
    Returns:
    --------
    DataFrame with aggregated feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model doesn't have feature_importances_")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    if categorical_mappings:
        # Aggregate one-hot encoded features
        aggregated = {}
        
        for feat in importance_df['feature']:
            # Check if it's a one-hot encoded feature
            is_encoded = False
            for cat_col in categorical_mappings.keys():
                if feat.startswith(f"{cat_col}_"):
                    if cat_col not in aggregated:
                        aggregated[cat_col] = 0
                    aggregated[cat_col] += importance_df[importance_df['feature'] == feat]['importance'].values[0]
                    is_encoded = True
                    break
            
            if not is_encoded:
                # Regular feature
                aggregated[feat] = importance_df[importance_df['feature'] == feat]['importance'].values[0]
        
        # Create new dataframe
        importance_df = pd.DataFrame(
            list(aggregated.items()),
            columns=['feature', 'importance']
        )
    
    return importance_df.sort_values('importance', ascending=False)


# Example usage for different model types
def get_model_specific_features(df: pd.DataFrame,
                               model_name: str,
                               categorical_cols: List[str],
                               numeric_cols: List[str]) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Get properly encoded features for a specific model type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    model_name : str
        Name of the model
    categorical_cols : List[str]
        Categorical column names
    numeric_cols : List[str]
        Numeric column names
        
    Returns:
    --------
    Tuple of (transformed_df, preprocessor)
    """
    # Determine model type
    tree_models = ['RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees', 
                   'GradientBoosting', 'HistGradientBoosting']
    linear_models = ['Ridge', 'Lasso', 'ElasticNet', 'LinearRegression', 
                     'LogisticRegression', 'SGD']
    other_models = ['SVR', 'SVC', 'Huber', 'KNeighbors']
    
    if model_name in tree_models:
        model_type = 'tree'
    elif model_name in linear_models or model_name in other_models:
        model_type = 'linear'
    else:
        # Default to linear (safer with one-hot encoding)
        model_type = 'linear'
    
    # Create handler
    handler = CategoricalFeatureHandler(
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols
    )
    
    # Create preprocessor
    preprocessor = handler.create_preprocessor(model_type)
    
    # Transform data
    X_transformed = preprocessor.fit_transform(df)
    feature_names = handler.get_feature_names(preprocessor, model_type)
    
    # Create dataframe
    X_df = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    
    return X_df, preprocessor