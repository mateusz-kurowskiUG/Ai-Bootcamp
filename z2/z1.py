from sklearn.preprocessing import StandardScaler
import numpy as np
from rich import print
import polars as pl

from mlp import Model


def make_bogus_data():
    arr_len = 2000
    tmp = np.random.randint(-7, 13, arr_len)
    rain_fall = np.random.uniform(0, 51, arr_len)
    cloudiness = np.random.rand(arr_len)
    matrix = np.array([tmp, rain_fall, cloudiness])
    df = pl.from_numpy(matrix, schema=["temp", "rain", "cloud"], orient="col")
    df.write_csv("./dataset.csv")


def read_csv() -> pl.DataFrame:
    df = pl.read_csv("./dataset.csv")
    print(df)
    return df


def standarize(df: pl.DataFrame) -> pl.DataFrame:
    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(df)
    scaled_df = pl.DataFrame(scaled_arr, ["tmp", "rain", "cloud"], orient="row")
    return scaled_df


if __name__ == "__main__":
    bogus = make_bogus_data()
    df = read_csv()
    standarized = standarize(df)
    tensors = 1

    model = Model()
    model.forward(tensors)
