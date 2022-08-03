#%%
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('data.csv')


# %%

plt.scatter(df.a, df.b)

# %%
