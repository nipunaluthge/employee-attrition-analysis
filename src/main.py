import pandas as pd
from src.data_preprocessing.preprocessing import preprocess

def main():
    df = pd.read_csv('data/original_data.csv')
    print("Original data shape:", df.shape)

    # Preprocess data: cleaning, encoding, scaling, outlier handling
    df_preprocessed = preprocess(df)
    # Save preprocessed data for reference
    df_preprocessed.to_csv("data/processed_attrition_data.csv", index=False)
    print("Preprocessed data saved to data/processed_attrition_data.csv")

if __name__ == '__main__':
    main()