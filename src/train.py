import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from data_prep import preprocess_data

# Load data
df = pd.read_csv("../data/student-mat.csv")  # Adjust path if necessary
preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)

# Fit the preprocessor and transform features
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_transformed, y_train)

# Evaluate
predictions = model.predict(X_test_transformed)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Save model and preprocessor to 'artifacts/' 
joblib.dump(model, "../artifacts/model.pkl")
joblib.dump(preprocessor, "../artifacts/preprocessor.pkl")

