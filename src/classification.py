import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

SEED = 42

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'data', 'creditcard.csv')
df = pd.read_csv(file_path)

feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:, 30: ].columns

data_features = df[feature_names]
data_target = df[target]

scaler = StandardScaler()
data_features_scaled = scaler.fit_transform(data_features)

X_train, X_test, y_train, y_test = train_test_split(data_features_scaled, data_target, train_size=0.80, test_size=0.20, random_state=SEED)

# Initialize Base Models
log_reg = LogisticRegression(random_state=SEED)
gbc = GradientBoostingClassifier(random_state=SEED)
rfc = RandomForestClassifier(random_state=SEED)

# Train Base Models
log_reg.fit(X_train, y_train.values.ravel())
gbc.fit(X_train, y_train.values.ravel())
rfc.fit(X_train, y_train.values.ravel())

# Generate Predictions for AdaBoost (Stacking)
train_preds = pd.DataFrame({
    "log_reg": log_reg.predict(X_train),
    "gbc": gbc.predict(X_train),
    "rfc": rfc.predict(X_train)
})

test_preds = pd.DataFrame({
    "log_reg": log_reg.predict(X_test),
    "gbc": gbc.predict(X_test),
    "rfc": rfc.predict(X_test)
})

# Train AdaBoost on Stacked Predictions
ada = AdaBoostClassifier(random_state=SEED)
ada.fit(train_preds, y_train.values.ravel())

# Final Predictions
final_pred = ada.predict(test_preds)

# Evaluate Performance
f1 = round(f1_score(y_test, final_pred), 2)
recall = round(recall_score(y_test, final_pred), 2)
precision = round(precision_score(y_test, final_pred), 2)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
