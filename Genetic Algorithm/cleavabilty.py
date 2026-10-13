import joblib
import os

model_dir = ".venv/lib/saved_models"

lr_model = joblib.load(
    os.path.join(model_dir, "logistic_regression.pkl")
)

svm_model = joblib.load(
    os.path.join(model_dir, "svm.pkl")
)

dt_model = joblib.load(
    os.path.join(model_dir, "decision_tree.pkl")
)

rf_model = joblib.load(
    os.path.join(model_dir, "random_forest.pkl")
)

knn_model = joblib.load(
    os.path.join(model_dir, "knn.pkl")
)

imputer = joblib.load(
    os.path.join(model_dir, "imputer.pkl")
)

label_encoder = joblib.load(
    os.path.join(model_dir, "label_encoder.pkl")
)

scaler_lr = joblib.load(
    os.path.join(model_dir, "scaler_logistic_regression.pkl")
)

scaler_svm = joblib.load(
    os.path.join(model_dir, "scaler_svm.pkl")
)

scaler_knn = joblib.load(
    os.path.join(model_dir, "scaler_knn.pkl")
)

feature_names = joblib.load(
    os.path.join(model_dir, "feature_names.pkl")
)

print("All models loaded successfully.")
