import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("loan_data.csv")

print("First 5 rows")
print(df.head())

# Step 2: Separate Features and Target
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# Step 3: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Step 6: Prediction
y_prediction = model.predict(X_test)

# Step 7: Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_prediction))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_prediction))

print("\nClassification Report:\n", classification_report(y_test, y_prediction))
