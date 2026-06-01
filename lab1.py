import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('QtAgg')


print("--- Đang tải dữ liệu Titanic ---")
df = sns.load_dataset('titanic')

print("\n" + "="*40)
print("BÀI 1: KHÁM PHÁ DỮ LIỆU BAN ĐẦU")
print("="*40)


print("\n1. 10 dòng dữ liệu đầu tiên:")
print(df.head(10)) 


print(f"\n2. Kích thước dữ liệu:")
print(f"- Số lượng bản ghi (rows): {df.shape[0]}")
print(f"- Số lượng thuộc tính (columns): {df.shape[1]}")


print("\n3. Kiểu dữ liệu của từng cột:")
print(df.dtypes)


print("\n4. Thống kê cơ bản cho các thuộc tính số:")

stats = df.describe().T[['mean', 'min', 'max', 'std']]
print(stats)


print("\n5. Đang vẽ biểu đồ trực quan hóa...")


sns.set_theme(style="whitegrid")


num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols].hist(bins=20, figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle("Histogram cho các thuộc tính số", fontsize=16)
plt.tight_layout()
plt.show()


cat_cols = ['sex', 'class', 'embark_town'] 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, ax=axes[i], hue=col, legend=False, palette='Set2')
    axes[i].set_title(f'Bar chart: {col}')
    axes[i].set_ylabel('Số lượng')
plt.suptitle("Bar chart cho các thuộc tính phân loại", fontsize=16)
plt.tight_layout()
plt.show()


print("\n" + "="*40)
print("BÀI 2: LÀM SẠCH DỮ LIỆU")
print("="*40)


df_original = df.copy()


print("\n1. Số lượng dữ liệu bị thiếu ở mỗi cột:")
print(df.isnull().sum())


print("\n2. Thực hiện xử lý dữ liệu thiếu:")

df['age'] = df['age'].fillna(df['age'].median())
print("- Đã điền giá trị trung vị cho cột 'age'.")


df = df.dropna(subset=['embarked', 'embark_town'])
print("- Đã xóa các bản ghi bị thiếu thông tin 'embarked'.")


dup_count = df.duplicated().sum()
print(f"\n3. Kiểm tra trùng lặp:")
print(f"- Số bản ghi trùng lặp tìm thấy: {dup_count}")
if dup_count > 0:
    df = df.drop_duplicates()
    print("- Đã loại bỏ các bản ghi trùng lặp.")

print("\n4. Chuẩn hóa dữ liệu số (Standardization) cho cột 'fare'")
scaler = StandardScaler()

df['fare_scaled'] = scaler.fit_transform(df[['fare']])
print("- Đã hoàn tất chuẩn hóa cột 'fare'. Dữ liệu sau chuẩn hóa có Mean ~ 0 và Std ~ 1.")


print("\n5. Đang vẽ Boxplot so sánh...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))


sns.boxplot(y=df_original['fare'], ax=axes[0], color='lightcoral')
axes[0].set_title('Boxplot cột Fare (Dữ liệu gốc)')
axes[0].set_ylabel('Giá vé')

sns.boxplot(y=df['fare_scaled'], ax=axes[1], color='lightgreen')
axes[1].set_title('Boxplot cột Fare (Sau khi Standardization)')
axes[1].set_ylabel('Giá vé (Scaled)')

plt.tight_layout()
plt.show()