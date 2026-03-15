import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("iris.csv")

# Inputs and output
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Iris Dataset Accuracy:", accuracy)
print("Flower Classes:", data['species'].unique())