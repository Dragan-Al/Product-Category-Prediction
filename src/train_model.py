import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# Load prepared dataset
df = pd.read_csv("data/products_cleaned.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)


# Define features and target
X = df[
    [
        "product title",
        "title_length",
        "word_count",
        "digit_count",
        "special_char_count",
        "uppercase_count",
        "longest_word_length"
    ]
]

y = df["category label"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "title",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2
            ),
            "product title"
        ),
        (
            "numeric",
            MinMaxScaler(),
            [
                "title_length",
                "word_count",
                "digit_count",
                "special_char_count",
                "uppercase_count",
                "longest_word_length"
            ]
        )
    ]
)


# Create LinearSVC pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])


# Train model
pipeline.fit(X_train, y_train)

print("Model training completed.")


# Evaluate model
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Train final model on all data
pipeline.fit(X, y)

print("Final model training completed.")


# Create models folder
os.makedirs("models", exist_ok=True)


# Save model
model_path = "models/product_category_model.pkl"

joblib.dump(pipeline, model_path)

print("Model saved to:", model_path)