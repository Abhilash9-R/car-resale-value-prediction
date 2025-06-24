import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("car_resale_data.csv")

# Convert categorical columns to numbers
le_fuel = LabelEncoder()
df["Fuel_Type"] = le_fuel.fit_transform(df["Fuel_Type"])

le_trans = LabelEncoder()
df["Transmission"] = le_trans.fit_transform(df["Transmission"])

# Features and target
X = df[["Car_Model_Year", "Mileage_kmpl", "Kilometers_Driven", "Fuel_Type", "Transmission"]]
y = df["Resale_Value"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)