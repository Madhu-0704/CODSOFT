import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select better features
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# Convert Sex to numbers
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Fill missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Inputs and output
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Improved Model Accuracy:", accuracy)