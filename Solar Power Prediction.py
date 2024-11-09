# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the synthetic dataset into a DataFrame
data = {
    'sunlight_hours': [5, 6, 7, 8, 4, 9, 10, 12, 6, 11, 8, 9, 10, 7, 8, 11, 12, 9, 10, 8],
    'temperature': [25, 26, 28, 30, 24, 32, 35, 40, 28, 37, 33, 35, 36, 29, 31, 39, 42, 34, 37, 32],
    'humidity': [60, 55, 50, 45, 70, 40, 35, 30, 55, 38, 45, 40, 50, 65, 55, 30, 25, 45, 38, 50],
    'wind_speed': [3.0, 2.5, 4.0, 5.0, 1.5, 6.0, 5.5, 7.0, 3.5, 6.5, 4.0, 5.0, 4.5, 3.0, 4.0, 5.5, 7.0, 5.0, 5.0, 4.0],
    'cloud_cover': [20, 10, 15, 5, 30, 10, 5, 0, 20, 5, 10, 7, 8, 25, 12, 3, 0, 6, 5, 15],
    'precipitation': [0.0, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'panel_efficiency': [85, 90, 88, 92, 80, 95, 96, 97, 85, 93, 91, 94, 90, 82, 88, 95, 98, 92, 93, 87],
    'dust_level': [30, 25, 20, 18, 35, 15, 12, 10, 22, 14, 19, 16, 20, 28, 21, 11, 9, 17, 13, 23],
    'power_generated': [50, 60, 65, 70, 45, 80, 95, 110, 62, 100, 72, 85, 90, 68, 75, 105, 115, 85, 95, 72]
}

df = pd.DataFrame(data)

# Step 2: Preprocessing the data
# Separate features and target variable
X = df.drop('power_generated', axis=1)  # Features
y = df['power_generated']               # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data (feature scaling) using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate Mean Squared Error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

# Example output for model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
