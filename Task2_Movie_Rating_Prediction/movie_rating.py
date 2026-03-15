import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("movies.csv",encoding='latin1')

# Select useful columns
data = data[['Year', 'Duration', 'Votes', 'Rating','Genre']]

# Remove missing values
data = data.dropna()

# Clean Year
data['Year'] = data['Year'].astype(str).str.extract(r'(\d+)')
data['Year'] = pd.to_numeric(data['Year'])

# Clean Duration
data['Duration'] = data['Duration'].astype(str).str.extract(r'(\d+)')
data['Duration'] = pd.to_numeric(data['Duration'])

data['Votes'] = data['Votes'].astype(str).str.replace(',', '')
data['Votes'] = pd.to_numeric(data['Votes'])

data['Genre'] = data['Genre'].astype(str).astype('category').cat.codes

# Inputs and output
X = data[['Year', 'Duration', 'Votes','Genre']]
y = data['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)

print("Movie Rating Prediction Score:", round(score,3))