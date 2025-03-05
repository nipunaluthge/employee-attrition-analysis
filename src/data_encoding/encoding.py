import pandas as pd

def encode_categorical(df):
    """
    Encode categorical features:
      - Maps binary features to 0/1.
      - One-hot encodes multi-class categorical features.
    """
    # Convert target: Attrition to binary (Yes=1, No=0)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Binary encoding for Gender and OverTime
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    # One-hot encoding for multi-class categoricals
    categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    return df