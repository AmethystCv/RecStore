import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dataset import DatasetLoader
import torch
# 假设这是你的数据

index, offset = torch.load("/dev/shm/2021/fbgemm_t856_bs65536.pt")
data = index.numpy()
# 使用Counter来统计出现次数
data_counter = Counter(data)

# 计算每个出现次数的累积比例
total_count = len(data)
sorted_counts = sorted(data_counter.values(), reverse=True)
cumulative_frequencies = np.cumsum(sorted_counts) / total_count

# 分组数据，每个组包含相同的出现次数范围
group_size = 2  # 每组的出现次数范围
grouped_counts = [sum(sorted_counts[i:i+group_size]) for i in range(0, len(sorted_counts), group_size)]
grouped_cumulative = np.cumsum(grouped_counts) / total_count

# 提取分组的出现次数和对应的累积比例
grouped_appearances = list(range(1, len(grouped_counts) * group_size + 1, group_size))
grouped_cumulative_values = list(grouped_cumulative)

# 绘制分组后的累积分布曲线
plt.plot(grouped_appearances, grouped_cumulative_values, marker='o')
plt.xlabel('出现次数')
plt.ylabel('累积比例')
plt.title('出现次数与累积比例曲线（分组）')
plt.grid(True)
plt.show()
plt.savefig("fbgemm_t856_bs65536.png")