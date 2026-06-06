import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

sns.pairplot(
    df,
    hue="target",
    diag_kind="kde",
    palette="Set2"
)
plt.show()