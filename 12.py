# Movie Rating Prediction - CODSOFT Internship Task 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset with encoding to handle special characters
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

# Display original columns
print("Original Columns:", df.columns)

# Drop rows with missing target (Rating)
df = df.dropna(subset=['Rating'])

# Select relevant features
features = ['Genre', 'Director', 'Votes', 'Duration', 'Year']
df = df[features + ['Rating']]

# Drop rows with any missing values
df.dropna(inplace=True)

# Clean the 'Votes' column (remove commas and convert to integer)
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)
# Clean the 'Duration' column to extract numeric value
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(int)

# Encode categorical features
label_enc = LabelEncoder()
df['Genre'] = label_enc.fit_transform(df['Genre'])
df['Director'] = label_enc.fit_transform(df['Director'])

# Features and target
X = df.drop('Rating', axis=1)
y = df['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)
