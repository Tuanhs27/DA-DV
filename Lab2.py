import matplotlib
matplotlib.use('QtAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

df = sns.load_dataset('titanic')
df = df.dropna(subset=['age', 'fare', 'pclass'])
features = ['age', 'fare', 'pclass']

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram: {feature}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.kdeplot(df[feature], fill=True, color='lightgreen')
    plt.title(f'KDE: {feature}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=df[features])
plt.title('Boxplot: age, fare, pclass')
plt.show()

ticker = "AAPL"
stock = yf.download(ticker, start="2022-01-01", end="2024-01-01")

stock["MA50"] = stock["Close"].rolling(window=50).mean()
stock["MA200"] = stock["Close"].rolling(window=200).mean()

plt.figure(figsize=(12, 6))
plt.plot(stock.index, stock["Close"], label="Close", color="blue")
plt.plot(stock.index, stock["MA50"], label="MA50", color="orange")
plt.plot(stock.index, stock["MA200"], label="MA200", color="red")
plt.title(f"Gia co phieu {ticker} va Trung binh dong (2022-2024)")
plt.xlabel("Ngay")
plt.ylabel("Gia (USD)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['fare'], color='lightcoral')
plt.title('Outliers: Fare')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['age'], color='lightblue')
plt.title('Outliers: Age')
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap: Ma tran tuong quan")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="class", y="age", data=df, palette="muted")
plt.title("Bai 5: Violin Plot phan bo tuoi theo hang ve")
plt.show()