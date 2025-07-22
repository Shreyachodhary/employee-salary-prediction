import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load your dataset
df = pd.read_csv("data/employee_data.csv")

# Encode categorical variables
le_edu = LabelEncoder()
le_role = LabelEncoder()
df["Education"] = le_edu.fit_transform(df["Education"])
df["JobRole"] = le_role.fit_transform(df["JobRole"])

# Features and target
X = df[["Experience", "Education", "JobRole"]]
y = df["Salary"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/salary_predictor.pkl")
joblib.dump(le_edu, "model/education_encoder.pkl")
joblib.dump(le_role, "model/jobrole_encoder.pkl")

print("✅ Model and encoders saved in the 'model/' folder.")
