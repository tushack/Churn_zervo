# =====================================================
# 1. IMPORTS (ZERVE SAFE)
# =====================================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# =====================================================
# 2. LOAD DATA
# =====================================================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =====================================================
# 3. DATA CLEANING & PREPROCESSING
# =====================================================
# Drop ID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing numeric values
df.fillna(df.median(numeric_only=True), inplace=True)

# Target encoding
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# 5. MODEL DEFINITION (BUSINESS OPTIMIZED)
# =====================================================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=6,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# =====================================================
# 6. CROSS VALIDATION (PROOF IT WORKS)
# =====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="roc_auc"
)

print("Cross-Validation ROC-AUC scores:", cv_auc)
print("Mean CV ROC-AUC:", np.mean(cv_auc))

# =====================================================
# 7. TRAIN FINAL MODEL
# =====================================================
model.fit(X_train, y_train)

# =====================================================
# 8. THRESHOLD TUNING (BUSINESS LOGIC)
# =====================================================
y_prob = model.predict_proba(X_test)[:, 1]

# Lower threshold to catch more churners
threshold = 0.40
y_pred = (y_prob >= threshold).astype(int)

# =====================================================
# 9. EVALUATION METRICS
# =====================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Test Set Metrics ===")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("ROC-AUC  :", roc_auc)

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# 10. FEATURE IMPORTANCE (EXPLAINABILITY)
# =====================================================
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# =====================================================
# 11. SAVE ARTIFACTS (PRODUCTION READY)
# =====================================================
joblib.dump(model, "churn_model_zerve.pkl")
feature_importance.to_csv("feature_importance.csv", index=False)

print("\n✅ Model and artifacts saved successfully")
import os
print(os.listdir())

