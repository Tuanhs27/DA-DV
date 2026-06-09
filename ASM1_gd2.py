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

# Đọc dữ liệu và xử lý cơ bản
df = pd.read_csv('data.csv')
df['avg_session_minutes'] = df['avg_session_minutes'].fillna(0)
df = df.dropna(subset=['user_id'])
df = df.drop_duplicates()
df = df[df['avg_session_minutes'] >= 0]

# Tối ưu hiệu năng: Lấy mẫu 10,000 dòng để chạy Machine Learning mượt mà
df_cluster = df.sample(n=10000, random_state=42) if len(df) > 10000 else df

print("="*60)
print("BAT DAU GIAI DOAN 2: PHAN CUM VA TRUC QUAN HOA NANG CAO")
print("="*60)

# ---------------------------------------------------------
# YÊU CẦU 1: PHÂN TÍCH CÁC MỐI QUAN HỆ ĐA BIẾN
# ---------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

# 1. Thời gian học và completion rate
sns.scatterplot(data=df_cluster, x='avg_session_minutes', y='completion_rate', alpha=0.5, ax=axes1[0], color='blue')
axes1[0].set_title('Thoi gian hoc & Ty le hoan thanh')

# 2. Số video xem và khả năng mua khóa học (future_purchase)
sns.boxplot(data=df_cluster, x='future_purchase', y='videos_watched', ax=axes1[1], palette='Set2')
axes1[1].set_title('So video xem & Mua khoa hoc')

# 3. Hoạt động AI recommendation vs enrollment
sns.scatterplot(data=df_cluster, x='ai_recommend_enroll', y='courses_enrolled', alpha=0.5, ax=axes1[2], color='green')
axes1[2].set_title('Goi y AI & Dang ky khoa hoc')

plt.tight_layout()
plt.show(block=False)

# ---------------------------------------------------------
# YÊU CẦU 2: PHÂN CỤM NGƯỜI DÙNG BẰNG K-MEANS
# ---------------------------------------------------------
features = ['avg_session_minutes', 'videos_watched', 'quizzes_taken', 'completion_rate', 'sessions_per_week']
X = df_cluster[features].fillna(0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng K-Means với 4 cụm theo chuẩn đề bài
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

# Lấy tọa độ tâm cụm và đặt tên nhóm
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features)

# Gắn nhãn 4 nhóm người dùng theo đúng mô tả của FPT Polytechnic
cluster_names = ['Power Learners', 'Casual Learners', 'Certificate Hunters', 'Passive Users']
cluster_df['Cluster_Name'] = cluster_names

print("\nĐã phân cụm thành công 4 nhóm:")
for idx, name in enumerate(cluster_names):
    count = len(df_cluster[df_cluster['Cluster'] == idx])
    print(f"- {name}: {count} người dùng.")

# ---------------------------------------------------------
# YÊU CẦU 3 & 4: TREEMAP, DENDROGRAM VÀ STAR GLYPHS (RADAR)
# ---------------------------------------------------------

# Biểu đồ Treemap: Thể hiện cấu trúc tỷ trọng các nhóm
plt.figure(figsize=(10, 6))
sizes = df_cluster['Cluster'].value_counts().sort_index().values
squarify.plot(sizes=sizes, label=cluster_names, alpha=0.8, color=sns.color_palette("pastel", 4))
plt.title("Treemap: Phan bo cau truc nhom nguoi dung")
plt.axis('off')
plt.show(block=False)

# Biểu đồ Dendrogram: Thể hiện khoảng cách và cấu trúc rẽ nhánh (Ward method)
plt.figure(figsize=(12, 6))
shc.dendrogram(shc.linkage(X_scaled[:500], method='ward'))
plt.title("Dendrogram: Cay phan cap cau truc hanh vi")
plt.xlabel("Nguoi dung (Index)")
plt.ylabel("Khoang cach")
plt.show(block=False)

# Biểu đồ Star Glyphs (Radar Chart): Biểu diễn hành vi đặc trưng của từng nhóm
N = len(features)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig4, ax4 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax4.set_theta_offset(np.pi / 2)
ax4.set_theta_direction(-1)
plt.xticks(angles[:-1], features)
ax4.set_rlabel_position(0)

colors = ['blue', 'orange', 'green', 'red']
for i in range(4):
    values = cluster_df.loc[i].drop('Cluster_Name').values.flatten().tolist()
    values += values[:1]
    # Chuẩn hóa về thang [0, 1] để đồ thị không bị tràn tỷ lệ
    max_val = max(values) if max(values) > 0 else 1
    normalized_values = [v/max_val for v in values]
    
    ax4.plot(angles, normalized_values, linewidth=2, linestyle='solid', label=cluster_names[i], color=colors[i])
    ax4.fill(angles, normalized_values, alpha=0.1, color=colors[i])

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title("Star Glyphs (Radar Chart): Bieu dien hanh vi 4 nhom")
plt.show()