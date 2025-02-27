import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# 读取Excel文件
df = pd.read_excel('data.xlsx', engine='openpyxl')

# 假设'Year'列为时间序列，'Scope_1_Total'为我们需要预测的碳排放量
df['Year'] = pd.to_datetime(df['Year'], format='%Y')  # 确保'Year'列是日期时间格式
carbon_emission = df.set_index('Year')['Scope_1_Total']

# 对 carbon_emission 序列进行一阶差分
carbon_emission_diff = carbon_emission.diff()
carbon_emission_diff = carbon_emission_diff.dropna()
carbon_emission_diff2 = carbon_emission_diff.diff()
carbon_emission_diff2 = carbon_emission_diff2.dropna()
# 进行ADF检验
result0 = adfuller(carbon_emission)
print('ADF Statistic: {}'.format(result0[0]))
print('p-value: {}'.format(result0[1]))
print('Critical Values:')
for key, value in result0[4].items():
    print('\t{}: {}'.format(key, value))

result1 = adfuller(carbon_emission_diff)
print('ADF Statistic: {}'.format(result1[0]))
print('p-value: {}'.format(result1[1]))
print('Critical Values:')
for key, value in result1[4].items():
    print('\t{}: {}'.format(key, value))

result2 = adfuller(carbon_emission_diff2)
print('ADF Statistic: {}'.format(result2[0]))
print('p-value: {}'.format(result2[1]))
print('Critical Values:')
for key, value in result2[4].items():
    print('\t{}: {}'.format(key, value))

plot_acf(carbon_emission_diff2)  # 以第一列数据为例，分析ACF
plt.show()
plot_pacf(carbon_emission_diff2)  # 分析PACF
plt.show()

# 拟合ARIMA模型
p = 1
d = 2
q = 0
model = ARIMA(carbon_emission, order=(p, d, q))  # 需要根据实际情况确定p,d,q的值
model_fit = model.fit()

print(model_fit.summary())


# 预测未来几年的碳排放量，比如预测未来5年
forecast_years = 5
forecast = model_fit.forecast(steps=forecast_years)

# 输出预测结果
print(forecast)

# 绘制实际值与预测值的对比图（如果需要）
plt.figure(figsize=(12, 6))
plt.plot(carbon_emission, label='Actual')
plt.plot(pd.date_range(carbon_emission.index[-1] + pd.DateOffset(years=1), periods=forecast_years+1, freq='YS'), forecast, marker='o', linestyle='-', color='red', label='Forecast')
plt.title('Carbon Emission Forecast')
plt.xlabel('Year')
plt.ylabel('Emission')
plt.legend()
plt.show()