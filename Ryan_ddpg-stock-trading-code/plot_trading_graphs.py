import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('C:\Github\Senior_Sem\Ryan_ddpg-stock-trading-code')

df = pd.read_csv('ddpg_cem_test_results.csv')

df_test = pd.read_hdf('./data/poloniex_30m.hf',key='test')

dash = df_test.DASHBTC.close.where((df_test.index.hour == 23) & (df_test.index.minute == 30)).dropna()[0:128]
ltc = df_test.LTCBTC.close.where((df_test.index.hour == 23) & (df_test.index.minute == 30)).dropna()[0:128] 
xmr = df_test.XMRBTC.close.where((df_test.index.hour == 23) & (df_test.index.minute == 30)).dropna()[0:128] 

dash = dash - dash[0] + 1
ltc = ltc - ltc[0] + 1
xmr = xmr - xmr[0] + 1

r = df['reward']
cum_r = [1]
for i in range(len(r)-1):
    cum_r.append(cum_r[-1] + (r[i]/100))



plt.plot(df['steps'], cum_r, label='CEMRL (DDPG)')
plt.plot(df['steps'], dash, label='DASHBTC')
plt.plot(df['steps'], ltc, label='LTCBTC')
plt.plot(df['steps'], xmr, label='XMRBTC')
plt.legend()
plt.title('CEMRL (DDPG) Trained')
plt.xlabel('Days')
plt.ylabel('Value')
# plt.show()
plt.savefig('trained_ddpgcem.png')


