import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("creditcard.csv")

# Inputs and output
X = data[['Time', 'Amount']]
y = data['Class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Fraud Detection Accuracy:", accuracy)