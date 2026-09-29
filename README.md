# Product Category Prediction

This project predicts the category of a product based on its title.

## Project Description

The goal of this project is to automatically predict the correct product category using machine learning.

The project uses more than 30,000 products.

## Project Structure

Product-Category-Prediction

data

products.csv

products_cleaned.csv

models

product_category_model.pkl

notebooks

product_category_prediction.ipynb

src

train_model.py

predict_category.py

README.md

## Dataset

The dataset contains product information such as Product ID, Product Title, Merchant ID, Category Label, Product Code, Number of Views, Merchant Rating and Listing Date.

The main feature used for prediction is the product title.

## Data Preparation

The data was cleaned and prepared in the Jupyter Notebook.

Additional features were created from the product title.

title_length

word_count

digit_count

special_char_count

uppercase_count

longest_word_length

The cleaned dataset was saved as:

data/products_cleaned.csv

## Machine Learning

Four models were tested.

Logistic Regression

LinearSVC

Multinomial Naive Bayes

Random Forest

The results were:

LinearSVC: 96.75%

Random Forest: 96.46%

Logistic Regression: 96.02%

Multinomial Naive Bayes: 94.12%

LinearSVC was selected as the final model.

## Evaluation

The models were evaluated using accuracy, precision, recall, F1-score and confusion matrix.

The complete evaluation can be found in the Jupyter Notebook.

## How to Install

Clone the repository and open the project folder.

Install the required packages:

pip install pandas scikit-learn joblib matplotlib

## Train the Model

Run:

python src/train_model.py

The trained model will be saved in:

models/product_category_model.pkl

## Predict a Category

Run:

python src/predict_category.py

Enter a product title.

Example:

iphone 7 32gb gold

The model will return the predicted category.

To stop the program, enter:

exit

## Technologies

Python

Pandas

Scikit-learn

Joblib

Matplotlib

Jupyter Notebook