# 加载数据看看
import numpy as np
data = np.load('data/electricity.npy')
customer_1 = data[0]

# 看不同时间段的统计
for i in range(0, len(customer_1), 10000):
    window = customer_1[i:i+1000]
    print(f"t={i}: mean={window.mean():.4f}, std={window.std():.4f}")
