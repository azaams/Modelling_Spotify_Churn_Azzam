import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print("File not found.")
        return None

def preprocess_data(df, target_column):
    """Cleaning and preprocessing the data."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # handling duplicate rows
    df = df.drop_duplicates()

    # handling missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])  # fill with mode for categorical
        else:
            df[col] = df[col].fillna(df[col].median())  # fill with median for numerical

    # encoding categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # separating features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y