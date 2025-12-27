import joblib
import pandas as pd

# Load model & features
model = joblib.load("churn_model_zerve.pkl")
model_features = joblib.load("model_features.pkl")

def churn_prediction_api(input_json: dict):
    """
    Zerve-compatible API-like churn prediction
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_json])

    # Add missing columns
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[model_features]

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(pred),
        "churn_probability": round(float(prob), 3)
    }


# ===== TEST CALL (FOR SCREENSHOT) =====
sample_input = {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 840,
    "SeniorCitizen": 0,
    "Contract_One year": 1,
    "InternetService_Fiber optic": 1,
    "PaymentMethod_Electronic check": 1,
    "PaperlessBilling_Yes": 1
}


response = churn_prediction_api(sample_input)
print("API Response:", response)
