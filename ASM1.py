import matplotlib
matplotlib.use('QtAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv('data.csv')

print(f"- Nguồn dữ liệu: Nền tảng học trực tuyến LearnX")
print(f"- Số lượng bản ghi (rows): {df.shape[0]}")
print(f"- Số lượng thuộc tính (columns): {df.shape[1]}")
print("- Các thuộc tính chính:")
print(df.info())

print("Số lượng missing values mỗi cột:\n", df.isnull().sum())

df['avg_session_minutes'] = df['avg_session_minutes'].fillna(0)
df = df.dropna(subset=['user_id'])

dup_count = df.duplicated().sum()
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"Đã xóa {dup_count} bản ghi trùng lặp.")

df = df[df['avg_session_minutes'] >= 0]
print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")

df_plot = df.sample(n=100000, random_state=42) if len(df) > 100000 else df

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.histplot(df_plot['avg_session_minutes'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Phân phối thời gian học (phút/phiên)')

sns.countplot(data=df_plot, x='sessions_per_week', ax=axes[0, 1], palette='viridis', hue='sessions_per_week', legend=False)
axes[0, 1].set_title('Số lần truy cập mỗi tuần')

sns.histplot(df_plot['completion_rate'], bins=20, kde=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Mức độ hoàn thành khóa học')

trend_data = df_plot.groupby('signup_days_ago')['avg_session_minutes'].mean().reset_index()
sns.lineplot(data=trend_data, x='signup_days_ago', y='avg_session_minutes', ax=axes[1, 1], color='coral')
axes[1, 1].set_title('Xu hướng thời gian học theo số ngày đăng ký')
axes[1, 1].invert_xaxis() 

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df_plot['avg_session_minutes'], ax=axes[0], color='lightblue')
axes[0].set_title('Outliers: Thời gian học')

sns.boxplot(y=df_plot['courses_enrolled'], ax=axes[1], color='lightgreen')
axes[1].set_title('Outliers: Số khóa đăng ký')

sns.boxplot(y=df_plot['total_spent_usd'], ax=axes[2], color='lightcoral')
axes[2].set_title('Outliers: Chi tiêu (USD)')
plt.tight_layout()
plt.show()

threshold_study = df['avg_session_minutes'].quantile(0.99)
hardcore_learners = df[df['avg_session_minutes'] > threshold_study]
print(f"- Có {len(hardcore_learners)} người dùng học cực kỳ nhiều (>{threshold_study:.2f} phút/phiên).")

ghost_users = df[(df['courses_enrolled'] >= 3) & (df['avg_session_minutes'] == 0)]
print(f"- Có {len(ghost_users)} người dùng đăng ký nhiều nhưng không học.")

threshold_spend = df['total_spent_usd'].quantile(0.99)
whales = df[df['total_spent_usd'] > threshold_spend]
print(f"- Có {len(whales)} người dùng chi tiêu cực cao (>{threshold_spend:.2f} USD).")