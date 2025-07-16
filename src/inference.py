import os
import joblib
import pandas as pd
from datetime import datetime
from src.data_loader import load_sample_data


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'models', 'model.joblib')

CSV_URL = (
    'https://drive.google.com/uc?id=1mZFlDYnk_MJkwdeia86GUXM6ZRU-H-C5'
)

PRED_PATH = 'predictions.csv'
REPORT_PATH = 'report.html'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)


X, _ = load_sample_data(CSV_URL)
X = X.head(5)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

preds = model.predict(X)
X['Predicted'] = preds
X.to_csv(PRED_PATH, index=False)
print(f"✅ Предсказания сохранены в {PRED_PATH}")

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Отчёт о предсказаниях</title>
</head>
<body>
    <h1>Предсказания кинетической энергии</h1>
    <p><strong>Дата:</strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Количество шагов:</strong> {len(preds)}</p>
    <p><strong>Предсказания:</strong></p>
    {X[['Predicted']].to_html(index=False)}
</body>
</html>
"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"📄 Отчёт сохранён в {REPORT_PATH}")
