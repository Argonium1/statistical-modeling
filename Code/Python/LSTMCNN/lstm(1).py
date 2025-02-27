import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()



# 转换为 PyTorch 的 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建 TensorDataset 和 DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = X_train.shape[2]
hidden_size = 128  # 隐藏层大小
num_layers = 2  # LSTM 层数
output_size = M  # 输出尺寸

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500
early_stopping_patience = 50
best_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor.to(device))
        val_loss = criterion(val_outputs, y_test_tensor.to(device))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.8f}, Val Loss: {val_loss.item():.8f}')

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print('Early stopping')
        break

# 加载最佳模型
model.load_state_dict(torch.load('best_lstm_model.pth'))

# 评估模型
model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor.to(device))
    mse_loss = nn.MSELoss()(predicted, y_test_tensor.to(device))
    print(f'Validation MSE: {mse_loss.item():.8f}')

    predicted = predicted.cpu().numpy()
    actual = y_test
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    print(f'Validation R²: {r2:.8f}')
    print(f'Validation MAE: {mae:.8f}')
