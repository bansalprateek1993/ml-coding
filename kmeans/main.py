import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kmeans import kMeansFromScratch

def load_dataset():
    df = load_iris(as_frame=True)
    return df.frame

def inspect_data(df):
    print(df.shape)
    print(df.describe())
    print(df.columns)

    print(df.isnull().sum())
    print(df.duplicated().sum())

def clean_data(df):
    print(df.duplicated().sum())
    drop = df.drop_duplicates()
    print(drop.duplicated().sum())
    return drop


if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    inspect_data(df)
    clean_data(df)

    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    
    # Model initialization
    model = kMeansFromScratch(n_cluster=3, max_iter=200, tolerance=1e-4, random_state=42)
    model.fit(df)
    labels = model.predict(df)
    print(labels)

    print("centroids =", model.centroids)
    
    print("intertia =", model.inertia(df))