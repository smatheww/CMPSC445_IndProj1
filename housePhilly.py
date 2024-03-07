import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
data = pd.read_csv('housePricePhilly.csv')

# Clean column names to remove leading/trailing spaces
data.columns = data.columns.str.strip()

# Data Preprocessing
# Convert SalePrice to numeric, removing non-numeric characters
# Updated to use a raw string for regex
data['SalePrice'] = data['SalePrice'].replace(r'[\$,]', '', regex=True).astype(float)

# Convert SaleDate to datetime format
data['SaleDate'] = pd.to_datetime(data['SaleDate'])

# Remove outliers based on IQR for key columns
key_columns = ['SalePrice', 'OpeningBid', 'SheriffCost', 'ZillowEstimate', 'RentEstimate']
for column in key_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

# Feature Engineering
data['PropertyAge'] = data['SaleDate'].dt.year - data['yearBuilt']
data['SizePerBedroom'] = data['finishedSqft'] / (data['bedrooms'] + 1)
data['SizePerBathroom'] = data['finishedSqft'] / (data['bathrooms'] + 1)
data['SqftAgeInteraction'] = data['finishedSqft'] * data['PropertyAge']

# Model Development
X = data[['OpeningBid', 'SheriffCost', 'ZillowEstimate', 'RentEstimate', 'PropertyAge', 'SizePerBedroom', 'SizePerBathroom', 'SqftAgeInteraction']]
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'MAE': mae, 'RMSE': rmse}

for result in results:
    print(f"{result} - MAE: {results[result]['MAE']}, RMSE: {results[result]['RMSE']}")
