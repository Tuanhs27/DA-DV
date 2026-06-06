import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
features = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium"]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

def star_plot(values, label):
    num_vars = len(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_title(label)
    plt.show()

for i in range(3):
    star_plot(scaled_data[i], f"Sample {i}")