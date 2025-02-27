import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False # 解决保存图像是负号'-'显示为方块的问题
# 读取CSV数据到DataFrame
data = pd.read_csv("F:\\Desktop\\2.csv")

# 选择指定年份的数据
years = [1997, 2010, 2021]
data_filtered = data[data["year"].isin(years)]

# 获取排放类别列表（假设从第 4 列开始）
emission_categories = list(data.columns[3:])  # 假设排放数据从第 4 列开始

# 获取堆叠数据
stacked_data = data_filtered.groupby("year")[emission_categories].sum()

# 创建堆叠柱状图
fig, ax = plt.subplots(figsize=(12, 10))  # 创建图的画布
stacked_data.plot(kind="bar", stacked=True, ax=ax)
plt.xlabel("年份")
plt.ylabel("总排放量（百万吨）")
plt.title("中国二氧化碳排放量堆叠柱状图（1997年、2010年、2021年）")
plt.xticks(rotation=0)

# 调整布局以解决警告
plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)  # 根据需要调整边距
plt.legend(title="排放类别", bbox_to_anchor=(-0.72, 1), loc='upper left', fontsize='small')

plt.show()


