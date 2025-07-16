import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.data_loader import load_sample_data

GOOGLE_DRIVE_FILE_ID = (
    '1mZFlDYnk_MJkwdeia86GUXM6ZRU-H-C5'
)
CSV_URL = (
    f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
)

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')

os.makedirs(MODEL_DIR, exist_ok=True)

X_train, X_test, y_train, y_test = load_sample_data(CSV_URL)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'✅ Accuracy: {accuracy:.4f}')

joblib.dump(model, MODEL_PATH)
print(f'✅ Model saved to {MODEL_PATH}')

print(model.feature_names_in_)