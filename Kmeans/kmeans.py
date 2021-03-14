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


data = pd.read_csv('data_by_year.csv')

newData = data[['energy', 'acousticness', 'danceability']]

x = np.array(newData)


X_scaled = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=10, max_iter=600).fit(X_scaled)


newData['kmeans'] = kmeans.labels_
newData.columns = ['energy', 'acousticness', 'danceability','kmeans']

print(newData)
fig = px.scatter_3d(newData, x='energy', y='acousticness', z='danceability',
              color='kmeans')
fig.show()

c0 = newData[newData['kmeans']==0]
c1 = newData[newData['kmeans']==1]
c2 = newData[newData['kmeans']==2]
c3 = newData[newData['kmeans']==3]
c4 = newData[newData['kmeans']==4]
c5 = newData[newData['kmeans']==5]
c6 = newData[newData['kmeans']==6]
c7 = newData[newData['kmeans']==7]
c8 = newData[newData['kmeans']==8]
c9 = newData[newData['kmeans']==9]

c0.drop(['kmeans'], axis=1, inplace=True)
c1.drop(['kmeans'], axis=1, inplace=True)
c2.drop(['kmeans'], axis=1, inplace=True)
c3.drop(['kmeans'], axis=1, inplace=True)
c4.drop(['kmeans'], axis=1, inplace=True)
c5.drop(['kmeans'], axis=1, inplace=True)
c6.drop(['kmeans'], axis=1, inplace=True)
c7.drop(['kmeans'], axis=1, inplace=True)
c8.drop(['kmeans'], axis=1, inplace=True)
c9.drop(['kmeans'], axis=1, inplace=True)


x = c0.values #returns a numpy array
c0_scaled = scaler.fit_transform(x)
c0 = pd.DataFrame(c0_scaled)
c0.columns = ['energy', 'acousticness', 'danceability' ]
c0=c0.melt(var_name='groups', value_name='vals')

x = c1.values #returns a numpy array
c1_scaled = scaler.fit_transform(x)
c1 = pd.DataFrame(c1_scaled)
c1.columns = ['energy', 'acousticness', 'danceability' ]
c1=c1.melt(var_name='groups', value_name='vals')

x = c2.values #returns a numpy array
c2_scaled = scaler.fit_transform(x)
c2 = pd.DataFrame(c2_scaled)
c2.columns = ['energy', 'acousticness', 'danceability']
c2=c2.melt(var_name='groups', value_name='vals')

x = c3.values #returns a numpy array
c3_scaled = scaler.fit_transform(x)
c3 = pd.DataFrame(c3_scaled)
c3.columns = ['energy', 'acousticness', 'danceability']
c3=c3.melt(var_name='groups', value_name='vals')

x = c4.values #returns a numpy array
c4_scaled = scaler.fit_transform(x)
c4 = pd.DataFrame(c4_scaled)
c4.columns = ['energy', 'acousticness', 'danceability']
c4=c4.melt(var_name='groups', value_name='vals')

x = c5.values #returns a numpy array
c5_scaled = scaler.fit_transform(x)
c5 = pd.DataFrame(c5_scaled)
c5.columns = ['energy', 'acousticness', 'danceability']
c5=c5.melt(var_name='groups', value_name='vals')

x = c6.values #returns a numpy array
c6_scaled = scaler.fit_transform(x)
c6 = pd.DataFrame(c6_scaled)
c6.columns = ['energy', 'acousticness', 'danceability']
c6=c6.melt(var_name='groups', value_name='vals')

x = c7.values #returns a numpy array
c7_scaled = scaler.fit_transform(x)
c7 = pd.DataFrame(c7_scaled)
c7.columns = ['energy', 'acousticness', 'danceability']
c7=c7.melt(var_name='groups', value_name='vals')

x = c8.values #returns a numpy array
c8_scaled = scaler.fit_transform(x)
c8 = pd.DataFrame(c8_scaled)
c8.columns = ['energy', 'acousticness', 'danceability']
c8=c8.melt(var_name='groups', value_name='vals')

x = c9.values #returns a numpy array
c9_scaled = scaler.fit_transform(x)
c9 = pd.DataFrame(c9_scaled)
c9.columns = ['energy', 'acousticness', 'danceability']
c9=c9.melt(var_name='groups', value_name='vals')


f, axes = plt.subplots(4, 1)
ax = sns.violinplot( data=c0 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])
ax = sns.violinplot( data=c2 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[2])
ax = sns.violinplot( data=c3 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c4 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c5 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c6 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c7 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c8 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])
ax = sns.violinplot( data=c9 ,x="groups", y="vals", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])

plt.show()