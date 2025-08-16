import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Paths
RAW_DATA_PATH = "data/raw/AmesHousing.csv"
PROCESSED_DIR = "data/processed"

# Make sure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)

# Drop IDs (not useful for prediction)
df = df.drop(columns=["Order", "PID"])

# Separate features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# Pipelines for preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Fit + transform
X_processed = preprocessor.fit_transform(X)

# Convert to DataFrame (OneHotEncoder returns sparse matrix → make dense)
X_processed = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Save processed datasets
X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

print("✅ Preprocessing complete! Processed data saved in 'data/processed/'")
