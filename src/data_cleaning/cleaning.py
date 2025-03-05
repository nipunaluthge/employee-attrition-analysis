import pandas as pd

def drop_constant_columns(df):
    """
    Drops constant or irrelevant columns if they exist.
    """
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    if existing_cols:
        df.drop(existing_cols, axis=1, inplace=True)
        print("Dropped columns:", existing_cols)
    else:
        print("No constant columns to drop.")
    return df