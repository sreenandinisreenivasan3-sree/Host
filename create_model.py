# create_model.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os

print("="*50)
print("Creating new model with exact versions...")
print("="*50)

# Check versions
import sklearn
import imblearn
import xgboost
print(f"sklearn version: {sklearn.__version__}")
print(f"imblearn version: {imblearn.__version__}")
print(f"xgboost version: {xgboost.__version__}")

# Load your data
print("\nLoading data...")
df = pd.read_csv("campus_placement_data.csv")
print(f"Data loaded: {df.shape}")

# Drop unnecessary columns
if 'student_id' in df.columns:
    df = df.drop('student_id', axis=1)
if 'salary_lpa' in df.columns:
    df = df.drop('salary_lpa', axis=1)

# Prepare features and target
X = df.drop('placed', axis=1)
y = df['placed']

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns ({len(num_cols)}): {num_cols[:5]}...")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Create preprocessor
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

# Create pipeline with simple parameters first
print("\nCreating pipeline...")
model = ImbPipeline([
    ("prep", preprocess),
    ("select", SelectKBest(score_func=f_classif, k=15)),  # Using 15 for stability
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1
    ))
])

# Train the model
print("Training model...")
model.fit(X, y)
print("Model trained successfully!")

# Test the model
print("\nTesting model...")
y_pred = model.predict(X[:5])
y_proba = model.predict_proba(X[:5])
print(f"Sample predictions: {y_pred}")
print(f"Sample probabilities: {y_proba[:, 1]}")

# Save the model
os.makedirs('model', exist_ok=True)
model_path = 'model/final_model.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Model saved to: {model_path}")
print(f"File size: {os.path.getsize(model_path)} bytes")

print("\n" + "="*50)
print("Model creation complete!")
print("="*50)