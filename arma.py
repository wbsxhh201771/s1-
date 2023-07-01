#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt
#参数初始化
discfile = 's13.xls'
forecastnum = 5

#读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile, index_col= '日期')
# 月度数据
data = data['指标值']
data = data.resample('D').mean().ffill()
 
# 可视化
data.plot(figsize=(12,4))
plt.title('co2 data')
plt.show()
res = adfuller(data)
print('p value:', res[1])
# # 一阶差分
# data_diff1 = data.diff()
# # 二阶差分
# data_diff2 = data_diff1.diff()
 
# 季节差分
data_diff = data.diff(2).dropna()
 
data_diff.plot(figsize=(12,4))
plt.title('co2 -  difference')
plt.show()
'''
# ADF检验
res = adfuller(data_diff)
print('p value:', res[1])
from statsmodels.tsa.stattools import arma_order_select_ic
bic_min_order = arma_order_select_ic(data_diff, max_ar=6, max_ma=4, ic='bic')['bic_min_order']
print(bic_min_order)
'''
from statsmodels.tsa.arima_model import ARMA
model = ARMA(data_diff, order=(1,0)).fit(disp=-1)
print(model.summary())
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axs = plt.subplots(2, 1)
plot_acf(data_diff, ax=axs[0])
plot_pacf(data_diff, ax=axs[1])
plt.show()
model2 = ARMA(data_diff, order=(1,0)).fit(disp=-1)
model2.conf_int() 
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.3)
 
model.resid.plot(ax=axs[0][0])
axs[0][0].set_title('residual')
 
model.resid.plot(kind='hist', ax=axs[0][1])
axs[0][1].set_title('histogram')
 
sm.qqplot(model.resid, line='45', fit=True, ax=axs[1][0])
axs[1][0].set_title('Normal Q-Q')
 
plot_acf(model.resid, ax=axs[1][1])
 
plt.show()
preds = model.predict(0, len(data_diff)+6)
 
# 也可只取未来预测值
# fcast = model2.forecast(6)
 
plt.figure(figsize=(12, 4))
data_diff.plot(color='g', label='data_diff')
preds.plot(color='r', label='predict')
plt.legend()
plt.show()
df1 = pd.DataFrame(data)
df2 = pd.DataFrame(preds, columns=['predict'])
df = pd.concat([df1, df2], axis=1)
df['result'] = df['predict'] + df['指标值'].shift(2)
 
plt.figure(figsize=(12, 4))
df['result'].plot(color='r', label='arma result')
df['指标值'].plot(color='g', label='co2 data')
plt.legend()
plt.show()
print(df['result'])
'''
import numpy as np
rmse=np.sqrt(mean_squared_error(list(df['指标值']),list(df['result'])))
print(rmse)
'''