import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('C:\Github\Senior_Sem\Ryan_ddpg-stock-trading-code')

df = pd.read_csv('ddpg_cem_test_results_untrained.csv')

df_test = pd.read_hdf('./data/poloniex_30m.hf',key='test')

x = df_test.DASHBTC.close.where((df_test.index.hour == 23) & (df_test.index.minute == 30)).dropna()[0:128]

r = df['reward']
cum_r = [1]
for i in range(len(r)-1):
    cum_r.append(cum_r[-1] + r[i])


plt.plot(df['steps'], cum_r, label='DDPG-CEMRL')
plt.plot(df['steps'], x, label='price1')
# plt.label()
plt.title('DDPG-CEMRL Untrained')
plt.show()


