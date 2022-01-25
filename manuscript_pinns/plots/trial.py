import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("case_2_test.dat", sep=" ", header=None, skiprows=[1])
print(df["y_true,"])