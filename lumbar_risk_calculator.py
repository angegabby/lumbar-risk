import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('/content/lumbar_fusion_cleaned.csv')
features = ['age', 'race_white', 'insurance_private', 'discharge_home', 
            'los_days', 'fusion_levels', 'anterior_approach', 'charlson_score', 
            'chf', 'smoking', 'obesity', 'icu_stay', 'steroid_use', 'ssi']
X = df[features].copy()
y = df['reoperation']

# Un-scale age and los_days
X.loc[:, 'age'] = X['age'] * 15 + 50
X.loc[:, 'los_days'] = X['los_days'] * 3 + 5

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.3, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# Apply original scaling
X_train.loc[:, 'age'] = (X_train['age'] - 50) / 15
X_train.loc[:, 'los_days'] = (X_train['los_days'] - 5) / 3
X_test.loc[:, 'age'] = (X_test['age'] - 50) / 15
X_test.loc[:, 'los_days'] = (X_test['los_days'] - 5) / 3

# Load and fit model
model = joblib.load('/content/lumbar_model.pkl')
print("✅ Model loaded!")
print("Fitting model...")
model.fit(X_train, y_train)

# Function to predict and calibrate risk
def predict_risk(age, race_white, insurance_private, discharge_home, los_days, fusion_levels, charlson_score, 
                 chf, smoking, obesity, icu_stay, steroid_use, ssi):
    scaled_age = (min(age, 75.5) - 50) / 15
    scaled_los = (min(los_days, 36.8) - 5) / 3
    data = pd.DataFrame({
        'age': [scaled_age], 'race_white': [int(race_white)], 'insurance_private': [int(insurance_private)],
        'discharge_home': [int(discharge_home)], 'los_days': [scaled_los], 'fusion_levels': [fusion_levels],
        'anterior_approach': [0], 'charlson_score': [charlson_score], 'chf': [int(chf)], 'smoking': [int(smoking)],
        'obesity': [int(obesity)], 'icu_stay': [int(icu_stay)], 'steroid_use': [int(steroid_use)], 'ssi': [int(ssi)]
    })
    probs = model.predict_proba(data)
    risk = probs[:, 1][0] * 100 if isinstance(probs, np.ndarray) else probs.iloc[:, 1][0] * 100
    risk_factor_count = sum([chf, smoking, obesity, icu_stay, steroid_use, ssi, charlson_score > 2, fusion_levels > 2, age > 60, los_days > 10])
    calibration_factor = 0.15 if risk_factor_count <= 2 else 0.8 if risk_factor_count <= 5 else 1.2
    calibrated_risk = min(risk * calibration_factor, 50.0)
    return data, calibrated_risk

# Test high-risk patient
print("\nTesting High-Risk Patient:")
data, risk = predict_risk(age=90, race_white=False, insurance_private=False, discharge_home=False, los_days=20, 
                          fusion_levels=4, charlson_score=4, chf=True, smoking=True, obesity=True, 
                          icu_stay=True, steroid_use=True, ssi=True)
print("\nScaled Inputs (High-Risk):")
print(f"Age (scaled): {(min(90, 75.5) - 50) / 15:.3f}")
print(f"Hospital Stay (scaled): {(min(20, 36.8) - 5) / 3:.3f}")
print(data)
print("\nHigh-Risk Reoperation Risk:")
print(f"Risk of Reoperation: {risk:.1f}%")
print("High Risk: Consult your surgeon for additional evaluation." if risk > 20 else 
      "Moderate Risk: Monitor closely with your healthcare provider." if risk > 10 else 
      "Low Risk: Continue with standard follow-up care.")

# Test low-risk patient
print("\nTesting Low-Risk Patient:")
data, risk = predict_risk(age=20, race_white=True, insurance_private=True, discharge_home=True, los_days=1, 
                          fusion_levels=1, charlson_score=0, chf=False, smoking=False, obesity=False, 
                          icu_stay=False, steroid_use=False, ssi=False)
print("\nScaled Inputs (Low-Risk):")
print(f"Age (scaled): {(min(20, 75.5) - 50) / 15:.3f}")
print(f"Hospital Stay (scaled): {(min(1, 36.8) - 5) / 3:.3f}")
print(data)
print("\nLow-Risk Reoperation Risk:")
print(f"Risk of Reoperation: {risk:.1f}%")
print("High Risk: Consult your surgeon for additional evaluation." if risk > 20 else 
      "Moderate Risk: Monitor closely with your healthcare provider." if risk > 10 else 
      "Low Risk: Continue with standard follow-up care.")
