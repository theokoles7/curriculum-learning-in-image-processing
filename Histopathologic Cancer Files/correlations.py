import pandas as pd
import numpy as np

df = pd.read_csv("cancer_test.csv")
RMSE = df["RMSE"]
Fourier = df["Fourier"]
WDH = df["WDH"]
WDV = df["WDV"]
WDD = df["WDD"]

print(np.corrcoef((RMSE, Fourier, WDH, WDV, WDD)))
