import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False # 解决保存图像是负号'-'显示为方块的问题
# 读取CSV文件
data = pd.read_csv('F:\\Desktop\\coal_oil_gas_vs_Year.csv')
data['all'] =data['raw_coal_nation']+data['natural_gas_nation']+data['raw_oil_nation']+data['Cement_nation']
# 设置绘图风格
plt.style.use('ggplot')

# 绘制折线图
plt.figure(figsize=(14, 7))

# 对于每个变量，我们绘制一条折线
plt.plot(data['year'], data['raw_coal_nation'], label='Raw Coal', marker='p')
plt.plot(data['year'], data['natural_gas_nation'], label='Natural Gas', marker='s')
plt.plot(data['year'], data['CO2_output'], label='CO2 Output', marker='^')
plt.plot(data['year'], data['raw_oil_nation'], label='raw_oil_nation', marker='o')
plt.plot(data['year'], data['Cement_nation'], label='Cement_nation', marker='d')


# 添加标题和标签
plt.title('Variables vs Year')
plt.xlabel('Year')
plt.ylabel('单位：Mt（百万吨）')
plt.legend()

# 显示图表
plt.grid(True)
plt.tight_layout()
plt.show()