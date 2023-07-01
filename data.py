import pandas as pd
discfile = 's1copy.xls'
from matplotlib import pyplot as plt
data = pd.read_excel(discfile)
q=data['指标值变更时间']
p=[]
for i in q:
    p.append(i[:10])
data['指标值变更时间']=p
#data.to_excel('s12.xls')
k=[]
t=0
data1 = pd.read_excel('s12.xls')
for i in range(2311):
    if p[i]!=p[i-1]:
        k.append(t)
        t=0
        t+=data1['指标值'][i]
    else:
        t+=data1['指标值'][i]
print(k)
print(len(k)) 
b=[]
for i in range(2311):
    if p[i]!=p[i-1]:
        b.append(p[i])
print(len(b))   
#plt.plot(b,k)
#plt.show()
print(b) 
data2 = pd.read_excel('s13.xls')
data2['指标值']=k
data2['指标值变更时间']=b
data2.to_excel('s13.xls')
