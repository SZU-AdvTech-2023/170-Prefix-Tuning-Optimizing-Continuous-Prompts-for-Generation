import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# 假设模型定义
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 设置设备为 cuda:0（可以根据实际情况选择不同的 GPU 设备）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 创建模型并使用 DataParallel 进行包装
model = DataParallel(SimpleModel(),device_ids=[0,1,2,3,4,5,6,7])
model = model.to(device)

# 生成随机输入数据
input_data = torch.randn(16, 10)
input_data = input_data.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 模型训练
for epoch in range(100):
    # 模拟数据加载
    # 注意：在实际应用中，你需要从数据加载器中获取真实的数据
    for _ in range(100000000000):
        # 模型前向传播
        output = model(input_data)

        # 模拟计算损失
        target = torch.randn_like(output)
        loss = criterion(output, target)

        # 模型反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 确保在使用 DataParallel 之后，模型参数被正确同步
print(model.module.fc.weight)
