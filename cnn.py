import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ====================== 超参数设置（可根据你的数据修改）======================
seq_len = 20       # 序列长度：必须是偶数（适配池化后seq_len//2，避免长度不一致）
input_size = 1     # 输入y的特征维度（单变量y=1，多变量对应修改）
output_size = 1    # 输出x的特征维度（单变量x=1，多变量对应修改）
# CNN专属超参数（可调整用于对比）
cnn_channels = 64  # 卷积核数量（对应LSTM的hidden_size，保持64方便对比）
kernel_size = 3    # 卷积核大小（时序序列常用3/5）
stride = 1         # 卷积步长
pool_kernel = 2    # 池化核大小
pool_stride = 2    # 池化步长
pool_padding = 0   # 池化padding（整数，替换原"same"）
batch_size = 32    # 批次大小
epochs = 100       # 训练轮次
learning_rate = 0.001  # 学习率
total_samples = 1000  # 总样本数

# 1. 定义exp(x)泰勒展开前m项和（m=ceil(x)）
def taylor_exp(x, max_terms=50):
    """计算exp(x)的泰勒展开前m=ceil(x)项和"""
    m = int(np.ceil(x))  # m=x向上取整
    s = 0.0
    term = 1.0  # k=0项: x^0/0!=1
    s += term
    for k in range(1, m+1):
        term *= x / k  # 递推计算x^k/k!，避免溢出
        s += term
    return s

# 2. 生成连续x轴（平滑序列，保证时序依赖）
x_full = np.linspace(0, 8, total_samples + seq_len)  # x∈[0,8]，exp(x)无溢出

# 3. 生成对应的y=泰勒展开前ceil(x)项和（函数项级数和）
y_full = np.array([taylor_exp(xi) for xi in x_full])

# 4. 滑动窗口构建序列样本
y_data = []
x_data = []
for i in range(total_samples):
    # 取seq_len长度的y序列作为输入
    y_seq = y_full[i:i+seq_len]
    # 取对应seq_len长度的x序列作为预测目标
    x_seq = x_full[i:i+seq_len]
    y_data.append(y_seq)
    x_data.append(x_seq)

# 5. 调整形状：(样本数, seq_len, input_size) → 原代码标准格式
y_data = np.array(y_data).reshape(-1, seq_len, input_size)
x_data = np.array(x_data).reshape(-1, seq_len, output_size)
# ==========================================================================
# 自动选择设备：有GPU用GPU，无GPU用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ---------------------- 数据预处理（和LSTM完全一致，保证对比公平）----------------------
# 归一化：CNN对数值范围也敏感，保持和LSTM相同的归一化策略
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_x = MinMaxScaler(feature_range=(0, 1))

y_scaled = scaler_y.fit_transform(y_data.reshape(-1, input_size)).reshape(y_data.shape)
x_scaled = scaler_x.fit_transform(x_data.reshape(-1, output_size)).reshape(x_data.shape)

# 划分训练集和测试集（8:2划分）
y_train, y_test, x_train, x_test = train_test_split(y_scaled, x_scaled, test_size=0.2, random_state=42)

# 转换成PyTorch张量，适配模型
y_train_tensor = torch.FloatTensor(y_train).to(device)
x_train_tensor = torch.FloatTensor(x_train).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)
x_test_tensor = torch.FloatTensor(x_test).to(device)

# 构建批量加载器，方便训练
train_dataset = TensorDataset(y_train_tensor, x_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(y_test_tensor, x_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- 【核心修改：修复MaxPool1d的padding参数】----------------------
class SimpleCNN(nn.Module):
    def __init__(self, input_size, seq_len, cnn_channels, kernel_size, output_size):
        super(SimpleCNN, self).__init__()
        # 1D卷积层：适配时序序列（seq_len为序列长度，input_size为特征维度）
        self.conv1 = nn.Conv1d(
            in_channels=input_size,    # 输入通道数=特征维度（单变量=1）
            out_channels=cnn_channels, # 卷积核数量（对应LSTM的hidden_size）
            kernel_size=kernel_size,   # 卷积核大小（滑动窗口提取局部时序特征）
            stride=stride,             # 步长
            padding="same"             # Conv1d支持"same"，保持序列长度不变
        )
        self.relu = nn.ReLU()  # 激活函数
        # 修复：MaxPool1d的padding改为整数（0），并显式指定步长
        self.pool = nn.MaxPool1d(
            kernel_size=pool_kernel,
            stride=pool_stride,
            padding=pool_padding  # 整数，替换原"same"
        )
        
        # 重新计算全连接层输入维度：池化后序列长度=seq_len//pool_stride（seq_len=20→10）
        pool_out_len = seq_len // pool_stride
        fc_input_dim = cnn_channels * pool_out_len
        self.fc1 = nn.Linear(fc_input_dim, 128)  # 中间全连接层
        self.fc2 = nn.Linear(128, seq_len * output_size)  # 输出序列长度和输入一致
    
    def forward(self, x):
        # CNN输入格式：(batch, channels, seq_len) → 需要转置原输入(batch, seq_len, input_size)
        x = x.transpose(1, 2)  # 转置后形状：(batch, input_size, seq_len)
        
        # 卷积+激活+池化
        x = self.conv1(x)      # 输出形状：(batch, cnn_channels, seq_len)
        x = self.relu(x)
        x = self.pool(x)       # 输出形状：(batch, cnn_channels, seq_len//2)
        
        # 展平：(batch, cnn_channels * seq_len//2)
        x = x.reshape(x.size(0), -1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)        # 输出形状：(batch, seq_len * output_size)
        
        # 重塑为目标形状：(batch, seq_len, output_size)
        x = x.reshape(x.size(0), seq_len, output_size)
        return x

# ---------------------- 初始化CNN模型（替换原LSTM初始化）----------------------
model = SimpleCNN(
    input_size=input_size,
    seq_len=seq_len,
    cnn_channels=cnn_channels,
    kernel_size=kernel_size,
    output_size=output_size
).to(device)

# 损失函数、优化器和LSTM完全一致（保证对比公平）
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ---------------------- 模型训练（和LSTM完全一致）----------------------
print("开始训练CNN模型...")
model.train()  # 开启训练模式
for epoch in range(epochs):
    train_loss = 0.0
    for batch_y, batch_x in train_loader:
        # 1. 前向传播：输入y，预测x
        outputs = model(batch_y)
        loss = criterion(outputs, batch_x)
        
        # 2. 反向传播与参数更新
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向计算梯度
        optimizer.step()       # 更新模型参数
        
        train_loss += loss.item() * batch_y.size(0)
    
    # 计算本轮平均训练损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # 每10轮打印一次损失，方便观察训练进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], 训练损失: {avg_train_loss:.6f}")

print("CNN训练完成！")

# ---------------------- 模型测试与评估（和LSTM完全一致）----------------------
model.eval()  # 开启评估模式
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():  # 测试时关闭梯度计算，节省内存
    for batch_y, batch_x in test_loader:
        outputs = model(batch_y)
        loss = criterion(outputs, batch_x)
        test_loss += loss.item() * batch_y.size(0)
        
        # 保存预测结果和真实值，后续反归一化
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_x.cpu().numpy())

# 计算测试集平均损失
avg_test_loss = test_loss / len(test_loader.dataset)
print(f"CNN测试集平均损失: {avg_test_loss:.6f}")

# 反归一化，恢复原始数值范围
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
predictions_original = scaler_x.inverse_transform(all_predictions.reshape(-1, output_size)).reshape(all_predictions.shape)
targets_original = scaler_x.inverse_transform(all_targets.reshape(-1, output_size)).reshape(all_targets.shape)

# ---------------------- 新y预测x的核心函数（和LSTM完全一致）----------------------
def predict_x(new_y):
    """
    输入新的y序列，预测对应的x序列
    :param new_y: 新的y数据，形状支持 (seq_len, input_size) 或 (n_samples, seq_len, input_size)
    :return: 预测的x序列，形状对应 (seq_len, output_size) 或 (n_samples, seq_len, output_size)
    """
    model.eval()
    # 处理单样本情况：(20,1) → (1,20,1)
    if len(new_y.shape) == 2:
        new_y = new_y.reshape(1, seq_len, input_size)
    
    # 1. 对新y做归一化
    new_y_reshaped = new_y.reshape(-1, input_size)
    new_y_scaled = scaler_y.transform(new_y_reshaped).reshape(new_y.shape)
    
    # 2. 转换成模型适配的张量
    new_y_tensor = torch.FloatTensor(new_y_scaled).to(device)
    
    # 3. 模型预测
    with torch.no_grad():
        pred_scaled = model(new_y_tensor)
    
    # 4. 反归一化，恢复原始数值
    pred_reshaped = pred_scaled.cpu().numpy().reshape(-1, output_size)
    pred_original = scaler_x.inverse_transform(pred_reshaped).reshape(pred_scaled.shape)
    
    # 还原单样本形状：(1,20,1) → (20,1)
    if pred_original.shape[0] == 1:
        pred_original = pred_original.reshape(seq_len, output_size)
    
    return pred_original

# ---------------------- 新数据预测（和LSTM完全一致）----------------------
test_samples = 100 
# 生成完整x序列（包含seq_len偏移，保证滑动窗口完整）
x_full_new = np.linspace(8.0, 10.0, test_samples + seq_len)
# 生成对应的y=泰勒展开值（和训练代码逻辑一致）
y_full_new = np.array([taylor_exp(xi) for xi in x_full_new])

# 滑动窗口构建序列样本（和训练代码完全一致）
y_data_test = []
x_data_test = []
for i in range(test_samples):
    y_seq = y_full_new[i:i+seq_len]
    x_seq = x_full_new[i:i+seq_len]
    y_data_test.append(y_seq)
    x_data_test.append(x_seq)

# 调整形状：(样本数, seq_len, 特征数)
y_data_test = np.array(y_data_test).reshape(-1, seq_len, input_size)
x_data_test = np.array(x_data_test).reshape(-1, seq_len, output_size)

# 预测对应的x
pred_x = predict_x(y_data_test)
print("\n新y预测x的示例结果：")
print(f"输入y的形状: {y_data_test.shape}")
print(f"预测x的形状: {pred_x.shape}")

# ---------------------- 结果可视化（和LSTM完全一致，方便对比）----------------------
# 1. 测试集样本预测效果对比
plt.figure(figsize=(10, 4))
plt.plot(targets_original[0, :, 0], label="real", color="blue")
plt.plot(predictions_original[0, :, 0], label="predict", color="red", linestyle="--")
plt.xlabel("time")
plt.ylabel("x")
plt.title("CNN - test set")
plt.legend()
plt.show()

# 2. 新样本预测效果对比
plt.figure(figsize=(10, 4))
plt.plot(x_data_test[0, :, 0], label="real", color="blue")
plt.plot(pred_x[0, :, 0], label="predict", color="red", linestyle="--")
plt.xlabel("sequence position")
plt.ylabel("x")
plt.title("CNN - new samples")
plt.legend()
plt.show()