import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
data_index = input("choose which data:(1,2,3): ")
io = r'./data_huya.csv'
data = pd.read_csv(io,sep = ',', header = 0, encoding = 'gbk')
# data = data.sample(frac=1).reset_index(drop=True)
# data = data.loc[:round(data.shape[0]/20)]
if data_index in ['3']:
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.loc[:round(data.shape[0]/5)]
else:
    data = data.sample(frac=0.4).reset_index(drop=True)
    data = data.loc[:round(data.shape[0] / 10)]

data['behavior'].fillna(0,inplace=True)
data['user_action'].fillna(0,inplace=True)
data1 = data[data['behavior'].isin([0])]
data2 = data[data['user_action'].isin([0])]
data3 = data[(data['behavior'].isin([0])) & (data['user_action'].isin([0]))]
data_set = [data1,data2,data3]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 填充空值
data = data_set[int(copy.copy(data_index))-1]
data.index = [i for i in range(data.shape[0])]
data['link_all_userip_city'].fillna(0,inplace=True)
data['link_all_mobile_city'].fillna(0,inplace=True)

ip_dict = {}
area_dict = {}
for i in range(data.shape[0]):
    if data.loc[i,'userip'] not in ip_dict:
        ip_dict[data.loc[i,'userip']] = [i]
    else:
        ip_dict[data.loc[i, 'userip']].append(i)
    area = str(data.loc[i,'link_all_userip_city']) + str(data.loc[i,'link_all_mobile_city'])
    if area in ['00']:
        continue
    if area not in area_dict:
        area_dict[area] = [i]
    else:
        area_dict[area].append(i)
# 按ip,地区建边
link = []
for ip in ip_dict.keys():
    if len(ip_dict[ip])>1:
        while ip_dict[ip]:
            index = ip_dict[ip].pop()
            for i in ip_dict[ip]:
                link.append([index,i])
            if len(ip_dict[ip]) in [1]:
                break
for area in area_dict.keys():
    if len(area_dict[area])>1:
        while area_dict[area]:
            index = area_dict[area].pop()
            for i in area_dict[area]:
                link.append([index,i])
            if len(area_dict[area]) in [1]:
                break
# # 去重
# link_n = []
# for i in link:
#     if i not in link_n:
#         link_n.append(i)
# link = link_n
if not os.path.isdir('./data/cora'):
    os.makedirs('./data/cora')
os.chdir(r'./data/cora')
with open("data{}_link.txt".format(data_index), 'w') as f:
    for i in link:
        f.write("%d\t%d\n"%(i[0],i[1]))
f.close()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# features
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# data = data3
# data.index = [i for i in range(data.shape[0])]
data['appver'].fillna(0,inplace=True)
data['appid'].fillna(0,inplace=True)
data['link_all_mobile_operators'].fillna(0,inplace=True)

appid_l = data['appid'].tolist()
appver_l = data['appver'].tolist()
terminal_l = data['link_all_mobile_operators'].tolist()
appid_ls = set(appid_l)
appver_ls = set(appver_l)
terminal_ls = set(terminal_l)
appid_index = dict(zip(appid_ls, [_ for _ in range(len(appid_ls))]))
appver_index = dict(zip(appver_ls, [_ for _ in range(len(appver_ls))]))
terminal_index = dict(zip(terminal_ls, [_ for _ in range(len(terminal_ls))]))

appid = [appid_index[i] for i in appid_l]
appver = [appver_index[i] for i in appver_l]
terminal = [terminal_index[i] for i in terminal_l]

# appid
embedding_1 = torch.nn.Embedding(100,10)
embedding_2 = torch.nn.Embedding(1000,10)
embedding_3 = torch.nn.Embedding(20,3)

embed_1 = embedding_1(torch.LongTensor(appid))
embed_2 = embedding_2(torch.LongTensor(appver))
embed_3 = embedding_3(torch.LongTensor(terminal))

embed = torch.cat([embed_1,embed_2,embed_3],dim=1).detach().numpy()
label = np.array([(1 if data.loc[i,'label_7']>0 else 0) for i in range(data.shape[0])]).reshape(data.shape[0],1)
index = np.array([i for i in range(data.shape[0])],dtype=str).reshape(data.shape[0],1)
feature = np.hstack((index,embed,label))

np.savetxt('data{}_feature.txt'.format(data_index),feature,fmt="%s")
print('over')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
