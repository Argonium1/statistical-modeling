import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# 读取数据
xls = pd.ExcelFile('data.xlsx')

carbon_emissions_data = []  # 存储各个工作表的 DataFrame
for sheet_name in xls.sheet_names:
    if sheet_name.startswith('19') or sheet_name.startswith('20'):  # 确保只读取年份相关的sheet
        df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
        carbon_emissions_data.append(np.array(df.values))

carbon_emissions_array = np.array(carbon_emissions_data)
print(carbon_emissions_array.shape)

N, M, S = carbon_emissions_array.shape[0], carbon_emissions_array.shape[1], carbon_emissions_array.shape[2]

# 将数据合并以进行标准化
all_data = carbon_emissions_array.reshape(-1, S)
scaler = StandardScaler()
scaled_all_data = scaler.fit_transform(all_data)
scaled_data_array = scaled_all_data.reshape(N, M, S)

# 将数据分为输入和输出
X = scaled_data_array[:, :, :-1]
y = scaled_data_array[:, :, -1]

# 划分训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# 定义LSTM-CNN混合模型
class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_channels, hidden_size, lstm_layers, output_size):
        super(LSTM_CNN_Model, self).__init__()

        # CNN部分
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM部分
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # CNN层
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        # 适应LSTM输入，调整维度
        x = x.permute(0, 2, 1)

        # LSTM层
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出作为序列的表示
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        x = nn.functional.relu(self.fc1(lstm_out))
        out = self.fc2(x)
        return out

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定模型参数
input_channels = X_train.shape[2]
hidden_size = 64
lstm_layers = 1
output_size = M

# 初始化模型并转移到指定设备
lstm_cnn_model = LSTM_CNN_Model(input_channels, hidden_size, lstm_layers, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(lstm_cnn_model.parameters(), lr=0.001)

# 将数据转换为 PyTorch 的 Tensor
X_train_tensor = torch.Tensor(X_train).permute(0, 2, 1).to(device)  # 调整维度
y_train_tensor = torch.Tensor(y_train).to(device)
X_test_tensor = torch.Tensor(X_test).permute(0, 2, 1).to(device)  # 调整维度
y_test_tensor = torch.Tensor(y_test).to(device)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 训练模型
num_epochs = 1000
early_stopping_patience = 100
best_loss = float('inf')
patience_counter = 0

# 训练循环
for epoch in range(num_epochs):
    lstm_cnn_model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = lstm_cnn_model(batch_X.to(device))
        loss = criterion(outputs, batch_y.to(device))
        loss.backward()
        optimizer.step()

    lstm_cnn_model.eval()
    with torch.no_grad():
        val_outputs = lstm_cnn_model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(lstm_cnn_model.state_dict(), 'best_lstm_cnn_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print('Early stopping')
        break


# 加载最佳模型
lstm_cnn_model.load_state_dict(torch.load('best_lstm_cnn_model.pth'))

# 评估模型
lstm_cnn_model.eval()
with torch.no_grad():
    test_outputs = lstm_cnn_model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.8f}")

    predicted = test_outputs.cpu().numpy()
    actual = y_test_tensor.cpu().numpy()
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    print(f'Validation R²: {r2:.8f}')
    print(f'Validation MAE: {mae:.8f}')

