import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/products.csv")

# Drop all rows with missing values
df = df.dropna(subset=["product title", "category label"])

# Clean product titles
df["product title"] = df["product title"].astype(str).str.strip()

# Standardize category labels
category_mapping = {
    "CPU": "CPUs",
    "fridge": "Fridges",
    "Mobile Phone": "Mobile Phones"
}

df["category label"] = df["category label"].replace(category_mapping)

# Create new feature with the length of each product title
df["title_length"] = df["product title"].str.len()

# Define features and label
X = df[["product title", "title_length"]]
y = df["category label"]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "product title"),
        ("length", MinMaxScaler(), ["title_length"])
    ]
)

# Define pipeline with the selected model
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to file
joblib.dump(pipeline, "models/product_category_model.pkl")

print("Model trained and saved as 'models/product_category_model.pkl'")