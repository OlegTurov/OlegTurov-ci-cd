import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.data_loader import load_sample_data

GOOGLE_DRIVE_FILE_ID = '1mZFlDYnk_MJkwdeia86GUXM6ZRU-H-C5'
CSV_URL = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')

os.makedirs(MODEL_DIR, exist_ok=True)

X_train, X_test, y_train, y_test = load_sample_data(CSV_URL)

model = LinearRegression(copy_X=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'✅ MSE: {mse:.4f}')
print(f'✅ R²: {r2:.4f}')

joblib.dump(model, MODEL_PATH)
print(f'✅ Model saved to {MODEL_PATH}')

print(model.feature_names_in_)



