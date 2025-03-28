import pandas as pd
from src.data_preprocessing.preprocessing import preprocess
from src.feature_selection.feature_selection import remove_highly_correlated_features
from src.data_analysis.modeling import train_and_evaluate
def main():
    df = pd.read_csv('data/original_data.csv')
    print("Original data shape:", df.shape)

    # Preprocess data: cleaning, encoding, scaling, outlier handling
    df_preprocessed = preprocess(df)
    # Save preprocessed data for reference
    df_preprocessed.to_csv("data/processed_attrition_data.csv", index=False)
    print("Preprocessed data saved to data/processed_attrition_data.csv")

    # Feature selection: remove highly correlated features
    df_reduced = remove_highly_correlated_features(df_preprocessed, threshold=0.9)
    print("Data shape after feature selection:", df_reduced.shape)

    # Model training, evaluation, and SHAP interpretability
    # eventhough fd has reduced, preprocessed df is used here because random forest and xgboost models can determine feature importance by their own
    train_and_evaluate(df_preprocessed)

if __name__ == '__main__':
    main()