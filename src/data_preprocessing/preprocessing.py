import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def handle_outliers_and_transform(df):
    """
    Handle outliers and reduce skewness by log-transforming MonthlyIncome.
    Optionally cap extreme values.
    """
    if 'MonthlyIncome' in df.columns:
        df['MonthlyIncomeLog'] = df['MonthlyIncome'].apply(lambda x: np.log1p(x))
        # Cap extreme values (optional)
        income_99 = df['MonthlyIncome'].quantile(0.99)
        df['MonthlyIncome_capped'] = df['MonthlyIncome'].clip(upper=income_99)
    return df

def scale_numeric_features(df):
    """
    Scales numeric features (except target and dummy variables).
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Select numeric columns: exclude target and dummy columns (which contain '_' in their name)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['Attrition'] and '_' not in col]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess(df):
    """
    Complete preprocessing pipeline: outlier handling, encoding, scaling.
    """
    # Drop constant columns first
    from src.data_cleaning.cleaning import drop_constant_columns
    df = drop_constant_columns(df)
    
    # Handle outliers and create log-transformed salary
    df = handle_outliers_and_transform(df)
    
    # Encode categorical variables
    from src.data_encoding.encoding import encode_categorical
    df = encode_categorical(df)
    
    # Scale numeric features
    df = scale_numeric_features(df)
    
    return df