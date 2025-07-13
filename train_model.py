import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Encode target
le = LabelEncoder()
train["Personality_encoded"] = le.fit_transform(train["Personality"])

# Prepare features
X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])

# Encode categorical columns
combined = pd.concat([X, X_test], axis=0)
cat_cols = combined.select_dtypes(include="object").columns.tolist()
encoder = OrdinalEncoder()
combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

X = combined.iloc[:len(X)]
X_test = combined.iloc[len(X):]

# Train XGBoost
model = xgb.XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    max_depth=4,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    num_class=4,  # Adjust based on your dataset
    random_state=42
)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "personality_model.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")
joblib.dump(le, "label_encoder.pkl")
