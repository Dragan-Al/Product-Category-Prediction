import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/product_category_model.pkl")

print("Product Category Prediction")
print("Enter 'exit' to stop.")


while True:
    product_title = input("\nEnter product title: ")

    if product_title.lower() == "exit":
        print("Program finished.")
        break

    product_title = product_title.strip()

    if not product_title:
        print("Please enter a product title.")
        continue

    # Create features
    data = pd.DataFrame({
        "product title": [product_title],
        "title_length": [len(product_title)],
        "word_count": [len(product_title.split())],
        "digit_count": [
            sum(char.isdigit() for char in product_title)
        ],
        "special_char_count": [
            sum(
                not char.isalnum() and not char.isspace()
                for char in product_title
            )
        ],
        "uppercase_count": [
            sum(char.isupper() for char in product_title)
        ],
        "longest_word_length": [
            max(
                [len(word) for word in product_title.split()],
                default=0
            )
        ]
    })

    # Predict category
    prediction = model.predict(data)

    print("Predicted category:", prediction[0])