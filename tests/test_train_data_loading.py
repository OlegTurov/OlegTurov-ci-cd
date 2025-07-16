from src.data_loader import load_sample_data


GOOGLE_DRIVE_FILE_ID = '1mZFlDYnk_MJkwdeia86GUXM6ZRU-H-C5'
CSV_URL = (
    f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
)


def test_data_loading_from_google_drive():
    X_train, X_test, y_train, y_test = load_sample_data(CSV_URL)

    assert X_train.shape[0] > 0, "Тренировочная выборка пуста"
    assert X_test.shape[0] > 0, "Тестовая выборка пуста"
    assert X_train.shape[1] == X_test.shape[1], (
        "Количество признаков в train и test не совпадает"
    )
    assert len(y_train) == X_train.shape[0], (
        "Размерности X_train и y_train не совпадают"
    )

    print(
        f"✅ Загружено: {X_train.shape[0]} train, "
        f"{X_test.shape[0]} test, {X_train.shape[1]} признаков"
    )
