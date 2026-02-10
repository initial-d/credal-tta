import yfinance as yf
import pandas as pd
import os

# 创建 data 目录
if not os.path.exists('data'):
    os.makedirs('data')

print("Downloading Real-World Financial Data...")

# ==========================================
# 1. S&P 500 (COVID Crash: March 2020)
# ==========================================
print("Fetching S&P 500 (^GSPC)...")
# 下载区间：2020-01-01 到 2020-06-30
sp500 = yf.download('^GSPC', start='2020-01-01', end='2020-06-30')

# 格式化：你的 loader 需要 'Date' 和 'Close'
sp500 = sp500[['Close']].reset_index()
sp500.columns = ['Date', 'Close'] # 重命名列以匹配 load_sp500_crisis

# 保存
sp500.to_csv('data/sp500.csv', index=False)
print(f"Saved data/sp500.csv ({len(sp500)} rows)")


# ==========================================
# 2. Bitcoin (Crash: May 2021)
# ==========================================
print("Fetching Bitcoin (BTC-USD)...")
# 下载区间：2021-04-01 到 2021-07-31
# 获取小时级数据需要短期，yfinance 免费版小时级有限制。
# 这里我们下载日线级别即可验证原理，或者尝试下载 90 天内的小时级（如果允许）。
# 为了稳妥，我们先下载日线，因为 Chronos 对日线处理得很好。
btc = yf.download('BTC-USD', start='2021-04-01', end='2021-07-31')

# 格式化：你的 loader 需要 'timestamp' 和 'close' (注意大小写)
btc = btc[['Close']].reset_index()
btc.columns = ['timestamp', 'close'] 

# 保存
btc.to_csv('data/bitcoin.csv', index=False)
print(f"Saved data/bitcoin.csv ({len(btc)} rows)")

print("\nDone! Now run your experiments.")

