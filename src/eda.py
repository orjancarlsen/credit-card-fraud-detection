import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'data', 'creditcard.csv')
df = pd.read_csv(file_path)

def plot_subplots(df: pd.DataFrame, scaled: bool = False) -> None:

    df_copy = df.copy()

    if scaled:
        scaler = StandardScaler()
        df_copy.iloc[:, 1:29] = scaler.fit_transform(df_copy.iloc[:, 1:29])

    plt.figure(figsize = (15, 12))
    plt.subplot(5, 6, 1) ; plt.plot(df_copy.V1) ; plt.subplot(5, 6, 15) ; plt.plot(df_copy.V15)
    plt.subplot(5, 6, 2) ; plt.plot(df_copy.V2) ; plt.subplot(5, 6, 16) ; plt.plot(df_copy.V16)
    plt.subplot(5, 6, 3) ; plt.plot(df_copy.V3) ; plt.subplot(5, 6, 17) ; plt.plot(df_copy.V17)
    plt.subplot(5, 6, 4) ; plt.plot(df_copy.V4) ; plt.subplot(5, 6, 18) ; plt.plot(df_copy.V18)
    plt.subplot(5, 6, 5) ; plt.plot(df_copy.V5) ; plt.subplot(5, 6, 19) ; plt.plot(df_copy.V19)
    plt.subplot(5, 6, 6) ; plt.plot(df_copy.V6) ; plt.subplot(5, 6, 20) ; plt.plot(df_copy.V20)
    plt.subplot(5, 6, 7) ; plt.plot(df_copy.V7) ; plt.subplot(5, 6, 21) ; plt.plot(df_copy.V21)
    plt.subplot(5, 6, 8) ; plt.plot(df_copy.V8) ; plt.subplot(5, 6, 22) ; plt.plot(df_copy.V22)
    plt.subplot(5, 6, 9) ; plt.plot(df_copy.V9) ; plt.subplot(5, 6, 23) ; plt.plot(df_copy.V23)
    plt.subplot(5, 6, 10) ; plt.plot(df_copy.V10) ; plt.subplot(5, 6, 24) ; plt.plot(df_copy.V24)
    plt.subplot(5, 6, 11) ; plt.plot(df_copy.V11) ; plt.subplot(5, 6, 25) ; plt.plot(df_copy.V25)
    plt.subplot(5, 6, 12) ; plt.plot(df_copy.V12) ; plt.subplot(5, 6, 26) ; plt.plot(df_copy.V26)
    plt.subplot(5, 6, 13) ; plt.plot(df_copy.V13) ; plt.subplot(5, 6, 27) ; plt.plot(df_copy.V27)
    plt.subplot(5, 6, 14) ; plt.plot(df_copy.V14) ; plt.subplot(5, 6, 28) ; plt.plot(df_copy.V28)
    plt.subplot(5, 6, 29) ; plt.plot(df_copy.Amount)
    plt.show()

if __name__ == '__main__':
    plot_subplots(df)
    plot_subplots(df, scaled=True)