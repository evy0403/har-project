import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help="Path to the training dataset")
parser.add_argument("--output_model", type=str, required=True, help="Path to save the trained model")

# For local testing, provide default values if arguments are not supplied
try:
    args = parser.parse_args()
except:
    args = argparse.Namespace(
        trainingdata="C:\\Users\\Evelyn\\Documents\\har-project\\dataset.xlsx",
        output_model="C:\\Users\\Evelyn\\Documents\\har-project\\trained_model"
    )

# Load the dataset
print(f"Loading dataset from: {args.trainingdata}")
df = pd.read_excel(args.trainingdata)
print("Dataset loaded successfully!")
print(df.head())

# Extract features (x, y, z) and target (activity)
X = df[['x-axis', 'y-axis', 'z-axis']].values
y = df['activity'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
print(f"Saving model to: {args.output_model}")
os.makedirs(args.output_model, exist_ok=True)  # Ensure the output folder exists
model_path = os.path.join(args.output_model, "model.pkl")
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")
