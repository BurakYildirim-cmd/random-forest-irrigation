import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def split_timewise(df):
    n = len(df)
    return (
        df.iloc[:int(n*0.7)],
        df.iloc[int(n*0.7):int(n*0.85)],
        df.iloc[int(n*0.85):]
    )