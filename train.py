# train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading Titanic dataset...")
df = pd.read_csv("train.csv")

# Clean the minimum features needed for a fast API
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features (X) and Target (y)
X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

# Train the model
print("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X, y)

# Save the model as a file
joblib.dump(model, "titanic_model.pkl")
print("Success! 'titanic_model.pkl' created and saved.")