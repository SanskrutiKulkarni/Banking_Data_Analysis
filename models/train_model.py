import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import pickle
import os

# ====================
# Load Data
# ====================
df = pd.read_csv("data/banking_data.csv")

# Target encoding
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# ====================
# Feature Engineering
# ====================
# Cyclical encoding for month
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
             'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
df['month_num'] = df['month'].map(month_map)
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
df.drop(columns=['month', 'month_num'], inplace=True)

# Interaction features
df['balance_per_age'] = df['balance'] / (df['age'] + 1)
df['is_default_and_loan'] = ((df['default'] == 'yes') & (df['loan'] == 'yes')).astype(int)

# Group job categories
job_groups = {
    'blue-collar': 'worker',
    'technician': 'worker',
    'services': 'service',
    'admin.': 'admin',
    'management': 'admin',
    'self-employed': 'entrepreneur',
    'entrepreneur': 'entrepreneur',
    'housemaid': 'other',
    'unemployed': 'other',
    'student': 'other',
    'retired': 'other',
    'unknown': 'other'
}
df['job'] = df['job'].map(job_groups)

# ====================
# Encode categorical
# ====================
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ====================
# Features & Target
# ====================
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ====================
# Model Training
# ====================
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos

models = [
    ("xgb", xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)),
    ("lgbm", lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42)),
    ("rf", RandomForestClassifier(class_weight='balanced', random_state=42)),
    ("logreg", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
]

# Train each model
for name, clf in models:
    clf.fit(X_train, y_train)

# ====================
# Model Stacking (Averaging Probabilities)
# ====================
train_probs = np.zeros(len(X_test))
for _, clf in models:
    train_probs += clf.predict_proba(X_test)[:, 1]
train_probs /= len(models)

# ====================
# Find Best Threshold for Balanced F1
# ====================
precisions, recalls, thresholds = precision_recall_curve(y_test, train_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best Threshold: {best_threshold:.2f}")
y_pred = (train_probs >= best_threshold).astype(int)

# ====================
# Evaluation
# ====================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ====================
# Save Model & Encoders
# ====================
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump({
        'models': models,
        'encoders': encoders,
        'feature_names': X.columns.tolist(),
        'threshold': best_threshold
    }, f)

print("\n✅ Stacked model trained and saved successfully!")
