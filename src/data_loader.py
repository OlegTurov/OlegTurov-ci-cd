import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    return pd.read_csv(path, delim_whitespace=True)

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df['PotEng'] = df['PotEng'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['KinEng'] = df['KinEng'].astype(float)

    X = df[['PotEng', 'Volume']]

    y = df['KinEng']

    return X, y

def load_and_preprocess(path: str):
    df = load_data(path)
    return preprocess_data(df)


def load_sample_data(path: str, test_size=0.2, random_state=42):
    X, y = load_and_preprocess(path)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)