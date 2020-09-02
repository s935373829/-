#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft Yahei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# # 1. 读取数据/数据类型整理

# In[4]:


data0=pd.read_excel(r'白酒价格.xlsx',sheet_name='底表').copy()


# In[5]:


data0.info()


# In[ ]:





# ### 鉴于分析目的以及数据的实际业务意义，我们这里选择销售、价格、品牌作为主要分析特征。

# In[6]:


data0['ID']=data0['ID'].astype(str)


# In[7]:


data0.dtypes


# # 2. 数据去重

# In[4]:


fr21=pd.DataFrame([[1,2,3,2,4],[1,3,4,3,4],[1,6,3,8,9],[1,3,4,3,4],[7,3,1,5,4]])
fr21


# In[5]:


get_ipython().run_line_magic('pinfo', 'fr21.drop_duplicates')


# In[6]:


fr21.drop_duplicates([0,4],keep='first')#第一个参数subset通过选取列确定判断重复的所在子集，第二个参数keep，确定保留重复数据的第一次出现，
#还是保留重复数据的最后一次出现，或者所有的重复数据包括源数据一起全部删掉。


# In[8]:


fr21.drop_duplicates([0,4],keep='last')


# In[9]:


fr21.drop_duplicates(keep='first')


# ### 回到项目中

# In[8]:


data0=data0.drop_duplicates(keep='first')


# In[9]:


data0.shape


# In[10]:


data0.info()


# ### data0的数据总共8133行，而我们的特征‘近30天销量’刚好8133行数据。这就说明该特征列没有空值数据，无需补值。
# 
# ### 但价格需要补值，这里我们采用就近补值

# # 3. 数据补值

# In[12]:


data0['价格（元）']=data0['价格（元）'].fillna(method='bfill')#就近补值


# In[13]:


data0.info()


# ### 逻辑值检验（此外还需对0价格进行补值修正）

# In[14]:


data0['价格（元）'].max(),data0['价格（元）'].min()


# In[15]:


data0.loc[data0['价格（元）']==0,'价格（元）']=int(data0['价格（元）'].mean())#把0价格补成平均值


# In[30]:


data0.loc[data0['价格（元）']==345,'价格（元）']


# In[36]:


print(data0['价格（元）'][1920])
data0['近30天销量（件）'].min()


# # 4. 接下来我们以每个月为统计周期对满足不同价格和销量的品牌数量分布图进行可视化。

# In[42]:


datalist=list(set(data0['日期' ].astype(str).values))#寻找统计时间点
print(sum([len(data0.loc[data0['日期' ]==daten,:]) for daten in datalist]))
datalist


# In[ ]:





# ## 这里我们可以任意挑选统计时间点

# In[43]:


data1=data0.loc[data0['日期' ]==datalist[6],:]
data1


# In[45]:


price_int=np.ceil((data1['价格（元）'].max()-data1['价格（元）'].min())/60)

price_int,data1['价格（元）'].max(),data1['价格（元）'].min()


# In[47]:


data_price_x=[(data1['价格（元）'].min()-1)+j*(price_int) for j in range(0,61)]
data_price_x


# In[58]:


sales_int=(data1['近30天销量（件）'].max()-data1['近30天销量（件）'].min())/15
sales_int,data1['近30天销量（件）'].max(),data1['近30天销量（件）'].min()


# In[59]:


data_sales_y=np.ceil([(data1['近30天销量（件）'].min()-1)+k*(sales_int+1) for k in range(0,16)]).tolist()
data_sales_y


# In[ ]:


#检查ID是否有重复


# In[60]:


def check_inf(data1,data_price_x1,data_sales_y1,i,j):
    dadir=data1[(data1['价格（元）']>data_price_x1[i])&(data1['价格（元）']<=data_price_x1[i+1])&(data1['近30天销量（件）']>data_sales_y1[j])&(data1['近30天销量（件）']<=data_sales_y1[j+1])]
    ser_id=dadir['ID'].value_counts()
    ser_id.name='IDNr'
    fr_ser=pd.DataFrame(ser_id)
    fr_ser['品牌']=[ list(set(data1.loc[data1['ID']==id,'品牌']))[0] for id in ser_id.index]
    return len(dadir),print(fr_ser)


# In[61]:


X1=data1.loc[(data1['价格（元）']>data_price_x[0])&(data1['价格（元）']<=data_price_x[1])& (data1['近30天销量（件）']>data_sales_y[0])&(data1['近30天销量（件）']<=data_sales_y[1]),:]
X2=X1['ID'].value_counts()
X2.name='IDNr'
X3=pd.DataFrame(X2)
X3['品牌']=[ list(set(data1.loc[data1['ID']==id,'品牌']))[0] for id in X2.index]
#list(set(data1.loc[data1['ID']=='522000491585','品牌']))[0]
X3


# In[62]:


#data
def plot_inf(data1,data_price_x2,data_sales_y2,j_sales):
    X=[];H=[];Info=[]
    for kz in range(60):
        h,info=check_inf(data1,data_price_x2,data_sales_y2,kz,j_sales)
        X.append(str(data_price_x2[kz])+'-'+str(data_price_x2[kz+1]))  # 价格分组
        H.append(h);Info.append(info)  # 酒的ID和品牌名，H是接收这个分组有多少数据的列表统计len
        Y=data_sales_y2[j_sales+1]    # 销售量分组定位
    return X,H,Y,Info


# In[64]:


fig=plt.figure(figsize=(18,15))
ax=Axes3D(fig)
cl_list=['#EE82EE','#EE4000','#EE0000','#DC143C','#DAA520','#8B2323','#8A2BE2','#87CEEB','#7D9EC0','#7CCD7C','#76EE00','#68228B','#4EEE94','#404040','#0000FF']
for sz in zip(range(15),cl_list):
    data_plot=plot_inf(data1,data_price_x,data_sales_y,sz[0])
    print(data_plot[0][:5],data_plot[1][:5],data_plot[2],sz[0])
    ax.bar(data_plot[0][:10],data_plot[1][:10],data_plot[2],zdir='y',color=sz[1])#ax.bar([列表，元素可以数字和字符串]，[只能是数值]，单个数值)
plt.xlabel('价格',fontsize=14)
plt.ylabel('30天销量',fontsize=14)
ax.set_zlabel('满足条件的销售记录',fontsize=14)
ax.set_title(str(datalist[8])+'销售记录数量分布图',fontsize=20)#这里必须修改统计时间点
plt.ylim([0,6000])
ax.view_init(elev=20,azim=50)


# #### 从上面的单月品牌数量分布直方图说明产生成交的品牌具有以下两个特征：
# #### 1.价格大约处在600元以下，尤其集中于300元以下。
# #### 2.月成交量大约在0-1800之间

# # 5. 行业大盘（2017-2018）表现，各价位段表现

# In[65]:


data0.groupby#第一个参数by 指认要分组列，第二个参数轴参数，确定按列还是按行，
#一般按列，也就是说axis=0.第三个参数是as_index,把我们的组标签当作索引返回结果。
#注意groupby适合离散数据分组
list(data0.groupby(by=['品牌']))[0][1]


# In[131]:


#[list(data0.groupby(by=['日期','品牌']))[j][1] for j in range(len(list(data0.groupby(by=['日期','品牌']))))]


# In[106]:


data0.groupby(by=['品牌'])['近30天销量'].sum()


# In[111]:


dfg,asx=plt.subplots(1,1,figsize=(12,10))
(data0.groupby('日期')['近30天销量'].sum()).plot(kind='line',ax=asx,c='r')
#data0.groupby('日期')['近30天销量'].sum()


# In[115]:


fig=plt.figure(figsize=(18,15))
ax=Axes3D(fig)
price_int1=(data0['价格'].max()-data0['价格'].min())/20
data_price_x1=np.ceil([(data0['价格'].min()-1)+j*(price_int1+1) for j in range(0,21)]).tolist() # 分成20组，价格
for prz in range(len(data_price_x1)-1):
    data_ind=data0.loc[(data0['价格']>data_price_x1[prz])&(data0['价格']<=data_price_x1[prz+1]),:] # 将数据中的价格分进对应的价格组，将整条数据加入进去
    ser_data=data_ind.groupby('日期')['近30天销量'].sum() # 按日期分组，将每月的所有数据的销量求和
    X_val,Sal_val=list(ser_data.index.astype(str)),ser_data.values
    ax.bar( X_val,Sal_val,data_price_x1[prz+1],zdir='y')
    print(data_price_x1[prz],data_price_x1[prz+1])
plt.xlabel('日期',fontsize=14)
plt.ylabel('价格',fontsize=14)
ax.set_zlabel('销量总合',fontsize=14)
ax.set_title('各价位段产品销量分布图',fontsize=20)
plt.ylim([0,6000])#限制价格范围，局部观察每个条形图系列
ax.view_init(elev=20,azim=50)


# ## 观察上面的线图和三维直方图可得下面的结论：
# ## 1. 全价位段行业大盘走势整体呈上升趋势。从第一个统计周期2017-05-24起开始缓慢增长至2017-08-15。然后进入快速增长阶段，然而在2017-11和2018-01两个统计周期分别下降到局部谷底值，
# 
# ## 随后急剧增长至峰值。
# 
# ## 2. 各价位段行业大盘销量变化趋势十分雷同于整体全价位段行业大盘销量变化趋势
# ## 3. 产品销量主要主要集中于4-580这个价位段。

# # 6.如何评价江小白品牌？

# In[138]:


dfg1,asx1=plt.subplots(1,1,figsize=(12,10))
data0.loc[data0['品牌']=='江小白',:].groupby('日期')['近30天销量'].sum().plot(kind='line',ax=asx1,c='r')
data0.loc[data0['品牌']=='江小白',:].groupby('日期')['近30天销量'].sum()


# #### 从第一个统计节点开始直至第二个统计节点，江小白销量急速下降，随后进入缓慢增长期直至统计节点2017-11-03.此后进入急剧增长期直至统计节点2018-02-27，达到峰值。

# In[126]:


#对江小白进行结构化分析
data2=data0.loc[data0['品牌']=='江小白',:].groupby(['日期','ID'])['近30天销量'].sum()
data2['2017-05-24'].sort_values()
#list(data0.loc[data0['品牌']=='江小白',:].groupby(['日期','ID']))


# In[149]:


data3=data0.loc[data0['品牌']=='江小白',:].groupby(['ID'])['近30天销量'].sum()
data3.sort_values(ascending=False)


# In[30]:


#for j in data2.index.levels[0]:
    #print(data2[j].sort_values(ascending=False))


# #### 江小白销量的前5分别是ID：536909908299、536872453030、536909880348、536871917851、554149816345
# 

# # 作业：请从中总结出他们的共性

# In[151]:


data3=(data0.groupby('品牌')['近30天销量'].sum()).sort_values(ascending=False)


# In[152]:


name_list=data3[data3>120000].index.tolist()
name_list


# In[153]:


dfg3,asx3=plt.subplots(1,1,figsize=(22,16))
for pn in name_list:
    data0[data0['品牌']==pn].groupby('日期')['近30天销量'].sum().plot(kind='line',ax=asx3,label=pn,legend=True)


# #### 从上图中我们发现各品牌销量的变化基本趋于一致，且整体均呈上升趋势。但是各品牌销量增长/减少幅度的变化以及销量水平
# #### 的差别。首先观察江小白销量发展趋势图，尽管其整体呈上升趋势，然而销量幅度变化却十分平缓且销售量一直在以上品牌中处于最低状态。与茅台相比，泸州老窖
# #### 在2018-02之前其销量趋势基本与茅台类似，仅整体销售量略低于茅台。然而在2018-02之后，泸州老窖在销量上出现拐点，茅台却大约从2018-01月始进入销量的高速增长期，直至2018年3月。洋河的销售趋势与茅台整体几乎相同，仅整体销量要略高于茅台。红星、牛栏山和五粮液三者变化趋势大致一致，但是红星在2018-02之前变化幅度十分平缓，整体无大幅度上下摆动，而五粮液与牛栏山的销量却一致处于大幅度摆动中。从销量水平上来看，五粮液最高，牛栏山居中，红星最低。
# 
# 

# In[ ]:





# In[ ]:




