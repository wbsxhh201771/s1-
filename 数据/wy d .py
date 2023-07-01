import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame,Series
plt.rcParams['font.sans-serif']=['FangSong']
data=pd.read_excel('网约车.xls')
data1=DataFrame(data,columns=['指标值'])
data2=data1['指标值']
x=range(2311)
plt.plot(x,data2)
plt.savefig("网约车.png",bbox_inches='tight')