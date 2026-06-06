import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["class"] = wine.target

features = [
    "alcohol",
    "malic_acid",
    "ash",
    "flavanoids",
    "proline"
]

df_plot = df[features + ["class"]]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_plot[features])

df_scaled = pd.DataFrame(scaled_features, columns=features)
df_scaled["class"] = df_plot["class"]

plt.figure(figsize=(10, 6))
parallel_coordinates(
    df_scaled,
    "class",
    colormap=plt.cm.Set1,
    linewidth=1
)
plt.title("Parallel Coordinates Plot Wine Dataset")
plt.xlabel("Features")
plt.ylabel("Normalized Value")
plt.show()