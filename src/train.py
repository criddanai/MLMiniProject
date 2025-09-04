import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Config
# -----------------------------
DATA_PATH = ("student_habits_performance.csv")  # ตรวจสอบชื่อไฟล์
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METADATA_DIR = "metadata"
FEATURE_STATS_PATH = os.path.join(METADATA_DIR, "feature_stats.json")

features = [
    'study_hours_per_day',
    'sleep_hours',
    'social_media_hours',
    'mental_health_rating',
    'attendance_percentage',
    'part_time_job'
]
target = 'exam_score'

# -----------------------------
# 2. Load & Split Data
# -----------------------------
def load_and_split_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: ไม่พบไฟล์ CSV ที่ {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print("Columns in CSV:", df.columns.tolist())

    # เลือกเฉพาะ features และ target
    X = df[features]
    y = df[target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train shape: {X_train.shape} Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# -----------------------------
# 3. Build Preprocess + Model
# -----------------------------
def build_pipeline():
    numeric_features = [
        'study_hours_per_day',
        'sleep_hours',
        'social_media_hours',
        'mental_health_rating',
        'attendance_percentage'
    ]
    categorical_features = ['part_time_job']

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # รองรับ sklearn เก่า/ใหม่
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])
    return model

# -----------------------------
# 4. Save Feature Stats
# -----------------------------
def save_feature_stats(X_train):
    stats = {"numeric": {}, "categorical": {}}

    for col in X_train.columns:
        if np.issubdtype(X_train[col].dtype, np.number):
            # numeric column -> แปลงเป็น float
            stats["numeric"][col] = {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std()),
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max())
            }
        else:
            # categorical column -> แปลงเป็น str
            stats["categorical"][col] = {
                "unique_values": [str(v) for v in X_train[col].unique()]
            }

    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(FEATURE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"✅ Feature stats saved as {FEATURE_STATS_PATH}")

# -----------------------------
# 5. Train, Evaluate, Save Model
# -----------------------------
def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = build_pipeline()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # RMSE รองรับ sklearn เก่า/ใหม่
    try:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"✅ RMSE: {rmse:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved as {MODEL_PATH}")

    # Save metadata
    save_feature_stats(X_train)

if __name__ == "__main__":
    main()
