import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

# 加载碳排放数据
emission_df = pd.read_excel('data.xlsx', index_col='Province')

# 加载距离矩阵
distance_df = pd.read_excel('distance.xlsx', index_col=0)

# 数据预处理：删除重复行（如果有），并标准化碳排放数据
emission_df.drop_duplicates(inplace=True)
scaler = StandardScaler()
emission_scaled = scaler.fit_transform(emission_df[['Raw coal', 'Crude oil', 'Natural gas', 'Cement']])

# 构建空间权重矩阵
# 这里假设距离越小，空间权重越大，可使用距离的倒数或e^(-distance)等方式调整
spatial_weight = 1 / (distance_df + 1e-10)  # 防止除以0，加一个小常数
spatial_weight_matrix = csr_matrix(spatial_weight.values)

# 空间加权的碳排放特征
weighted_emissions = spatial_weight_matrix.dot(emission_scaled)

# 选择聚类数量k，这里以5为例
k = 5

# 应用K-Means聚类
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(weighted_emissions)

# 添加聚类标签到原始数据框
emission_df['Cluster'] = cluster_labels

# 输出结果
print(emission_df.sort_values('Cluster'))