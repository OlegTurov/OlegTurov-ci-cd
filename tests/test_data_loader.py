import pandas as pd
from src.data_loader import preprocess_data


def test_preprocess_data():
    # Взял первые 6 значений
    data = {
        'Step': [0, 10, 20, 30, 40, 50],
        'PotEng': [
            -6.7733681, -6.8589894, -7.0603088,
            -7.3078383, -7.5870058, -7.7778348
        ],
        'KinEng': [
            0.1497, 0.12699745, 0.061868765,
            0.038348628, 0.1232391, 0.16965119
        ],
        'Volume': [
            592.27671, 583.32725, 559.47762,
            532.10578, 510.24857, 492.20321
        ]
    }
    df = pd.DataFrame(data)

    X, y = preprocess_data(df)

    assert isinstance(X, pd.DataFrame)
    assert X.shape == (6, 2)
    assert y.shape == (6,)