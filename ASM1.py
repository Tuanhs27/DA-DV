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

df['study_time'] = df['study_time'].fillna(0)
df = df.dropna(subset=['user_id'])

dup_count = df.duplicated().sum()
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"Đã xóa {dup_count} bản ghi trùng lặp.")

df = df[df['study_time'] >= 0]
print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.histplot(df['study_time'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Phân phối thời gian học')

sns.countplot(data=df, x='weekly_visits', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Số lần truy cập mỗi tuần')

sns.histplot(df['completion_rate'], bins=20, kde=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Mức độ hoàn thành khóa học (%)')

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    trend_data = df.groupby('date')['study_time'].sum().reset_index()
    sns.lineplot(data=trend_data, x='date', y='study_time', ax=axes[1, 1], color='coral')
    axes[1, 1].set_title('Xu hướng học theo thời gian')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df['study_time'], ax=axes[0], color='lightblue')
axes[0].set_title('Outliers: Thời gian học')

sns.boxplot(y=df['courses_enrolled'], ax=axes[1], color='lightgreen')
axes[1].set_title('Outliers: Số khóa đăng ký')

sns.boxplot(y=df['spending'], ax=axes[2], color='lightcoral')
axes[2].set_title('Outliers: Chi tiêu')
plt.tight_layout()
plt.show()

threshold_study = df['study_time'].quantile(0.99)
hardcore_learners = df[df['study_time'] > threshold_study]
print(f"- Có {len(hardcore_learners)} người dùng học cực kỳ nhiều (>{threshold_study} giờ).")

ghost_users = df[(df['courses_enrolled'] > 5) & (df['study_time'] == 0)]
print(f"- Có {len(ghost_users)} người dùng đăng ký nhiều nhưng không học.")

threshold_spend = df['spending'].quantile(0.99)
whales = df[df['spending'] > threshold_spend]
print(f"- Có {len(whales)} người dùng chi tiêu cực cao (bất thường).")