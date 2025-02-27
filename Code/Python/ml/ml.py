import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
data = pd.read_excel("data.xlsx")

# 假设 Scope_1_Total 之外的指标是 X，Scope_1_Total 是 Y
X = data.drop(['Scope_1_Total', 'Scope_2_Heat', 'Scope_2_Electricity', 'Other_Energy'], axis=1)
Y = data['Scope_1_Total']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 归一化处理
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化模型
linear_reg = LinearRegression()
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
gradient_boosting_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)

# 训练模型
linear_reg.fit(X_train_scaled, y_train)
random_forest_reg.fit(X_train_scaled, y_train)
gradient_boosting_reg.fit(X_train_scaled, y_train)

# 预测
y_pred_linear = linear_reg.predict(X_test_scaled)
y_pred_rf = random_forest_reg.predict(X_test_scaled)
y_pred_gb = gradient_boosting_reg.predict(X_test_scaled)

# 评估模型性能
print("线性回归 MSE:", mean_squared_error(y_test, y_pred_linear))
print("随机森林 MSE:", mean_squared_error(y_test, y_pred_rf))
print("梯度提升回归 MSE:", mean_squared_error(y_test, y_pred_gb))

print("线性回归 R^2:", r2_score(y_test, y_pred_linear))
print("随机森林 R^2:", r2_score(y_test, y_pred_rf))
print("梯度提升回归 R^2:", r2_score(y_test, y_pred_gb))