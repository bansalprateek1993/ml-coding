import numpy as np
import pandas as pd
from regression import LinearRegressionFromScratch

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1 — Get the dataset
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

# Step 2 — Understand the dataset
def inspect_data(df):
    # Shape
    print(df.shape)

    # data types
    print(df.dtypes)

    # Describe
    print(df.describe())

    # Missing values
    print(df.isnull().sum())

    # duplicates
    print(df.duplicated().sum())


# Step 4 — Remove duplicates
def clean_data(df):
    print(df.duplicated().sum())
    df = df.drop_duplicates()
    print(df.duplicated().sum())
    return df

# Step 5 — Separate X and y
def seperate_features_with_label(df):
    label_column = "MedHouseVal"
    X = df.drop(columns=[label_column])
    y = df[label_column]

    print(X.shape, y.shape)
    return X, y

# Step 6 - Divide it into train, test and val
def data_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = load_data()
    print("shape:", df.shape)
    print("columns:", df.columns)

    # Dirting the data - Step 3 — Let's deliberately create a dirty dataset
    df.loc[10:20, "AveRooms"] = np.nan
    df.loc[50:55, "Population"] = np.nan

    # Duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    inspect_data(df)

    df = clean_data(df)
    # inspect_data(df)

    X, y = seperate_features_with_label(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y)

    # Step 7 - Fill missing value
    train_median = X_train.median()

    X_train, X_val, X_test = X_train.fillna(train_median), X_val.fillna(train_median), X_test.fillna(train_median)

    print(X_train.shape, X_val.shape, X_test.shape)

    # Step 8 — Convert to NumPy
    X_train = X_train.to_numpy(dtype=np.float64)
    X_val = X_val.to_numpy(dtype=np.float64)
    X_test = X_test.to_numpy(dtype=np.float64)

    y_train = y_train.to_numpy(dtype=np.float64)
    y_val = y_val.to_numpy(dtype=np.float64)
    y_test = y_test.to_numpy(dtype=np.float64)

    # Feature scaling as we are going to use gradient descent
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Step 15 - Train
    model = LinearRegressionFromScratch(lr=0.01, epochs=2000)
    model.fit(X_train, y_train)

    # Step 16- validate
    val_pred = model.predict(X_val)
    print("val_loss mse", model.mse(y_val, val_pred))
    print("val_loss rmse", model.rmse(y_val, val_pred))
    print("val_loss r2", model.r2(y_val, val_pred))

    # Step 17 - Test
    test_pred = model.predict(X_test)
    print("test_loss mse", model.mse(y_test, test_pred))
    print("test_loss rmse", model.rmse(y_test, test_pred))
    print("test_loss r2", model.r2(y_test, test_pred))

    



