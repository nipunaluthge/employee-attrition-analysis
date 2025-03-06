import pandas as pd
import numpy as np
import itertools

def remove_highly_correlated_features(df, threshold=0.9):
    """
    Identify and remove redundant features based on a correlation threshold.
    Returns a new DataFrame with redundant features removed.
    """
    # Compute absolute correlation matrix
    corr_matrix = df.corr().abs()
    
    # Find pairs of features with correlation > threshold
    high_corr_pairs = []
    for feature1, feature2 in itertools.combinations(corr_matrix.columns, 2):
        corr_value = corr_matrix.loc[feature1, feature2]
        if corr_value > threshold:
            high_corr_pairs.append((feature1, feature2, corr_value))
    
    # Decide on features to remove:
    features_to_remove = []
    # Salary related: Keep 'MonthlyIncomeLog' and drop 'JobLevel', 'MonthlyIncome', etc.
    salary_related = ['JobLevel', 'MonthlyIncome', 'RoleAvgIncome', 'RelIncomeToRoleAvg']
    for col in salary_related:
        if col in df.columns:
            features_to_remove.append(col)
    
    # For department dummies: drop one redundant (e.g., drop 'Department_Sales' if exists)
    if 'Department_Sales' in df.columns and 'Department_Research & Development' in df.columns:
        features_to_remove.append('Department_Sales')
    
    # Drop job role interaction terms (columns with '_Income' in their name starting with 'JobRole_')
    job_role_income = [col for col in df.columns if col.startswith('JobRole_') and '_Income' in col]
    features_to_remove.extend(job_role_income)
    
    # Remove duplicates from the list
    features_to_remove = list(set(features_to_remove))
    print("Features to remove due to high correlation:", features_to_remove)
    
    # Drop the features if they exist
    df_reduced = df.drop(columns=[col for col in features_to_remove if col in df.columns])
    return df_reduced