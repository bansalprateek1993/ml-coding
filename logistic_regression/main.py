import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lr import lrFromScratch


def load_data():
    df = load_breast_cancer(as_frame=True)
    return df.frame

def inspect_data(df):
    print(df.shape)
    print(df.columns)

    print(df.describe())
    print(df.dtypes)

    print(df.isnull().sum())
    print(df.duplicated().sum())

def clean_data(df):
    print(df.duplicated().sum())
    drop = df.drop_duplicates()
    print(drop.duplicated().sum())
    return drop

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.columns)
    inspect_data(df)

    # Add dirty data
    df.loc[30:43, "smoothness error"] = np.nan
    df.loc[20:24, "mean texture"] = np.nan

    df = pd.concat([df, df.iloc[0:10]], ignore_index=True)
    inspect_data(df)

    df = clean_data(df)

    # Divide it into X and y
    X = df.drop(columns=["target"])
    y = df["target"]

    # Divide it into train, test, val
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Fill na values
    train_med = X_train.median()

    X_train = X_train.fillna(train_med)
    X_val = X_val.fillna(train_med)
    X_test = X_test.fillna(train_med)

    # Convert it into numpy
    X_train = X_train.to_numpy(dtype=np.float64)
    X_val = X_val.to_numpy(dtype=np.float64)
    X_test = X_test.to_numpy(dtype=np.float64)

    y_train = y_train.to_numpy(dtype=np.float64)
    y_val = y_val.to_numpy(dtype=np.float64)
    y_test = y_test.to_numpy(dtype=np.float64)

    # Scalar transofrm
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train
    model = lrFromScratch(lr=0.01, epochs=2000)
    model.fit(X_train, y_train)

    val_probs = model.predict_probs(X_val)
    val_pred = model.predict(X_val)

    print(y_val.size, val_pred.size)
    val_metrics = model.cls_metrics(y_val, val_pred)
    print(val_metrics)






