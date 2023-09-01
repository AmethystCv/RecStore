import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 创建自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :-1], dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)
        return features, label

# 创建自定义数据集对象
dataset = CustomDataset("data.csv")

# 创建数据加载器
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 迭代加载数据批次
for batch_features, batch_labels in dataloader:
    print("批次特征:")
    print(batch_features)
    print("批次标签:")
    print(batch_labels)
    print()
