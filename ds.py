import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# ----------------------
# 1. 判别器定义（LSTM）
# ----------------------
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时间步
        out = self.fc(out)
        return self.sigmoid(out)


# ----------------------
# 2. 计算 discriminative score
# ----------------------
def compute_discriminative_score(real_data, fake_data, epochs=20, batch_size=64, lr=1e-3):
    """
    real_data: numpy array [N, seq_len, dim]
    fake_data: numpy array [N, seq_len, dim]
    return: discriminative score (float)
    """

    # 数据准备
    X = np.concatenate([real_data, fake_data], axis=0)
    y = np.concatenate([np.ones(len(real_data)), np.zeros(len(fake_data))], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型
    input_dim = X.shape[2]
    model = Discriminator(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x).squeeze()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().numpy()
        y_pred_label = (y_pred > 0.5).astype(int)
        acc = np.mean(y_pred_label == y_test.numpy())

    # discriminative score
    disc_score = np.abs(acc - 0.5)
    return disc_score


# ----------------------
# 3. 示例
# ----------------------
if __name__ == "__main__":
    # 随机模拟数据
    real_data = np.random.randn(500, 24, 6)   # 500个真实序列，每个长度24，维度6
    fake_data = np.random.randn(500, 24, 6)   # 500个生成序列

    score = compute_discriminative_score(real_data, fake_data)
    print("Discriminative Score:", score)
