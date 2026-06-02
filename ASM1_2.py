import matplotlib
matplotlib.use('QtAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
import squarify
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

df = pd.read_csv('data.csv')

missing_counts = df.isnull().sum()
dup_count = df.duplicated().sum()

df['avg_session_minutes'] = df['avg_session_minutes'].fillna(0)
df = df.dropna(subset=['user_id'])
if dup_count > 0:
    df = df.drop_duplicates()
df = df[df['avg_session_minutes'] >= 0]

unique_count = len(df)
df_plot = df.sample(n=100000, random_state=42) if len(df) > 100000 else df
df_cluster = df.sample(n=10000, random_state=42) if len(df) > 10000 else df

threshold_study = df['avg_session_minutes'].quantile(0.99)
hardcore_learners = df[df['avg_session_minutes'] > threshold_study]
ghost_users = df[(df['courses_enrolled'] >= 3) & (df['avg_session_minutes'] == 0)]
threshold_spend = df['total_spent_usd'].quantile(0.99)
whales = df[df['total_spent_usd'] > threshold_spend]

print("="*60)
print("BAO CAO DU LIEU TU DONG")
print("="*60)
print(f"1. Kich thuoc du lieu chuan: {df.shape[0]} dong, {df.shape[1]} cot.")
print(f"2. Tong so gia tri thieu: {missing_counts.sum()}")
print(f"3. Tong so ban ghi trung lap: {dup_count}")

print("\n4. KHOANG GIA TRI BAT THUONG (IQR BOUNDS):")
cols_to_check = ['avg_session_minutes', 'courses_enrolled', 'total_spent_usd']
for col in cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"- {col}: Nguong bat thuong < {lower:.2f} hoac > {upper:.2f} (Phat hien: {len(outliers)} ban ghi)")

print("\n5. PHAT HIEN CAC HANH VI BAT THUONG:")
print(f"- nguoi dung hoc cuc ky nhieu (>{threshold_study:.2f} phut): {len(hardcore_learners)} nguoi")
print(f"- nguoi dung dang ky nhieu khoa nhung khong hoc: {len(ghost_users)} nguoi")
print(f"- nguoi dung chi tieu bat thuong (>{threshold_spend:.2f} USD): {len(whales)} nguoi")
print("="*60)

fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(y=missing_counts.index, x=missing_counts.values, ax=axes1[0], hue=missing_counts.index, legend=False, palette='Reds_r')
axes1[0].set_title('So luong gia tri thieu moi cot')
axes1[1].pie(
    [unique_count, dup_count] if dup_count > 0 else [unique_count], 
    labels=['Duy nhat', 'Trung lap'] if dup_count > 0 else ['Duy nhat (100%)'], 
    autopct='%1.4f%%' if dup_count > 0 else None, 
    colors=['#66b3ff', '#ff9999'], 
    startangle=90
)
axes1[1].set_title('Ty le ban ghi trung lap')
plt.tight_layout()
plt.show(block=False)

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
sns.histplot(df_plot['avg_session_minutes'], bins=30, kde=True, ax=axes2[0, 0], color='skyblue')
axes2[0, 0].set_title('Phan phoi thoi gian hoc (phut/phien)')
sns.countplot(data=df_plot, x='sessions_per_week', ax=axes2[0, 1], palette='viridis', hue='sessions_per_week', legend=False)
axes2[0, 1].set_title('So lan truy cap moi tuan')
sns.histplot(df_plot['completion_rate'], bins=20, kde=True, ax=axes2[1, 0], color='lightgreen')
axes2[1, 0].set_title('Muc do hoan thanh khoa hoc')
trend_data = df_plot.groupby('signup_days_ago')['avg_session_minutes'].mean().reset_index()
sns.lineplot(data=trend_data, x='signup_days_ago', y='avg_session_minutes', ax=axes2[1, 1], color='coral')
axes2[1, 1].set_title('Xu huong thoi gian hoc theo ngay')
axes2[1, 1].invert_xaxis()
plt.tight_layout()
plt.show(block=False)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df_plot['avg_session_minutes'], ax=axes3[0], color='lightblue')
axes3[0].set_title('Gia tri bat thuong: Thoi gian hoc')
sns.boxplot(y=df_plot['courses_enrolled'], ax=axes3[1], color='lightgreen')
axes3[1].set_title('Gia tri bat thuong: So khoa dang ky')
sns.boxplot(y=df_plot['total_spent_usd'], ax=axes3[2], color='lightcoral')
axes3[2].set_title('Gia tri bat thuong: Chi tieu (USD)')
plt.tight_layout()
plt.show(block=False)

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df_cluster, x='avg_session_minutes', y='completion_rate', alpha=0.5, ax=axes4[0], color='blue')
axes4[0].set_title('Thoi gian hoc & Ty le hoan thanh')
sns.boxplot(data=df_cluster, x='future_purchase', y='videos_watched', ax=axes4[1], palette='Set2', hue='future_purchase', legend=False)
axes4[1].set_title('So video xem & Mua khoa hoc')
sns.scatterplot(data=df_cluster, x='ai_recommend_enroll', y='courses_enrolled', alpha=0.5, ax=axes4[2], color='green')
axes4[2].set_title('Goi y AI & Dang ky')
plt.tight_layout()
plt.show(block=False)

features = ['avg_session_minutes', 'videos_watched', 'quizzes_taken', 'completion_rate', 'sessions_per_week']
X = df_cluster[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features)
cluster_names = ['Nhom Hoc Nhieu', 'Nhom Hoc It', 'Nhom San Chung Chi', 'Nhom It Hoat Dong']
cluster_df['Cluster_Name'] = cluster_names

plt.figure(figsize=(10, 6))
sizes = df_cluster['Cluster'].value_counts().sort_index().values
squarify.plot(sizes=sizes, label=cluster_names, alpha=0.8, color=sns.color_palette("pastel", 4))
plt.title("Treemap: Phan bo cac nhom nguoi dung")
plt.axis('off')
plt.show(block=False)

plt.figure(figsize=(12, 6))
shc.dendrogram(shc.linkage(X_scaled[:500], method='ward'))
plt.title("Dendrogram (Cau truc nhom hanh vi)")
plt.xlabel("Nguoi dung (Index)")
plt.ylabel("Khoang cach")
plt.show(block=False)

N = len(features)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig7, ax7 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax7.set_theta_offset(np.pi / 2)
ax7.set_theta_direction(-1)
plt.xticks(angles[:-1], features)
ax7.set_rlabel_position(0)

colors = ['blue', 'orange', 'green', 'red']
for i in range(4):
    values = cluster_df.loc[i].drop('Cluster_Name').values.flatten().tolist()
    values += values[:1]
    max_val = max(values) if max(values) > 0 else 1
    normalized_values = [v/max_val for v in values]
    ax7.plot(angles, normalized_values, linewidth=2, linestyle='solid', label=cluster_names[i], color=colors[i])
    ax7.fill(angles, normalized_values, alpha=0.1, color=colors[i])

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title("Bieu do Radar (Dac diem 4 nhom)")
plt.show()