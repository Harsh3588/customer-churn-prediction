import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    # Remove customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Encode categorical columns
    le = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])

    return df