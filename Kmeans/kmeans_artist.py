import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

scaler = MinMaxScaler()


data = pd.read_csv('data_by_artist.csv')

data.drop(['key', 'mode', 'count'], axis=1, inplace=True)
newData = data[['energy', 'acousticness', 'tempo']]

x = np.array(newData)


X_scaled = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=4, max_iter=600).fit(X_scaled)


newData['kmeans'] = kmeans.labels_
newData.columns = ['energy', 'acousticness', 'tempo','kmeans']

print(newData)
fig = px.scatter_3d(newData, x='energy', y='acousticness', z='tempo',
              color='kmeans')
fig.show()

c0 = newData[newData['kmeans']==0]
c1 = newData[newData['kmeans']==1]
c2 = newData[newData['kmeans']==2]
c3 = newData[newData['kmeans']==3]


c0.drop(['kmeans'], axis=1, inplace=True)
c1.drop(['kmeans'], axis=1, inplace=True)
c2.drop(['kmeans'], axis=1, inplace=True)
c3.drop(['kmeans'], axis=1, inplace=True)


x = c0.values #returns a numpy array
c0_scaled = scaler.fit_transform(x)
c0 = pd.DataFrame(c0_scaled)
c0.columns = ['energy', 'acousticness', 'tempo' ]
c0=c0.melt(var_name='groups', value_name='vals')

x = c1.values #returns a numpy array
c1_scaled = scaler.fit_transform(x)
c1 = pd.DataFrame(c1_scaled)
c1.columns = ['energy', 'acousticness', 'tempo' ]
c1=c1.melt(var_name='groups', value_name='vals')

x = c2.values #returns a numpy array
c2_scaled = scaler.fit_transform(x)
c2 = pd.DataFrame(c2_scaled)
c2.columns = ['energy', 'acousticness', 'tempo']
c2=c2.melt(var_name='groups', value_name='vals')

x = c3.values #returns a numpy array
c3_scaled = scaler.fit_transform(x)
c3 = pd.DataFrame(c3_scaled)
c3.columns = ['energy', 'acousticness', 'tempo']
c3=c3.melt(var_name='groups', value_name='vals')



f, axes = plt.subplots(4, 1)
ax = sns.violinplot( data=c0 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])
ax = sns.violinplot( data=c2 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[2])
ax = sns.violinplot( data=c3 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])

plt.show()