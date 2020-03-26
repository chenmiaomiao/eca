import numpy as np
import pandas as pd


def load_wis():
  df = pd.read_csv("breast-cancer-wisconsin.csv")
  df = df.drop(df.columns[-1], axis=1)
  df = df.drop("id", axis=1)

  df["diagnosis"] = df["diagnosis"].astype("category")
  cat_col = df.select_dtypes(["category"]).columns
  print(cat_col)
  df[cat_col] = df[cat_col].apply(lambda x: x.cat.codes)

  x_woid = df[df.columns[1:]]
  y_woid = df["diagnosis"]


  x = x_woid.to_numpy()
  y = y_woid.to_numpy()
  print(y)

  np.save("x-wis.npy", x)
  np.save("y-wis.npy", y)

  return x, y

if __name__ == '__main__':
  load_wis()