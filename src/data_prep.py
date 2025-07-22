import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def preprocess_data(df):
    # Create a new feature based on study intensity
    df["StudyIntensity"] = df["Hours Studied"] / (df["Sleep Hours"] + 1)

    # Separate features and target
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    # Identify numeric and categorical columns
    numeric_features = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced", "StudyIntensity"]
    categorical_features = ["Extracurricular Activities"]

    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Combine into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return preprocessor, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv("../data/student-mat.csv")
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
    joblib.dump(preprocessor, "../artifacts/preprocessor.pkl")
