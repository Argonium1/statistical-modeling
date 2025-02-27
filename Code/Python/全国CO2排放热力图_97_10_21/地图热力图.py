import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# 步骤1：加载数据
data = pd.read_csv('F:\\Desktop\\地图热力图.csv')

# 步骤2：加载中国地图数据
china_map = gpd.read_file('F:\\Desktop\\中华人民共和国.json')

# 合并 CO2 数据到 GeoDataFrame
merged_data = china_map.merge(data, how='left', left_on='name', right_on='locate')

# 绘制热力图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged_data.plot(column='Total_apparent_CO2_emissions_(mt)', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

# 设置图表标题和其他参数
ax.set_title('China CO2 Emissions by Province')
ax.set_axis_off()

# 显示图表
plt.show()