import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

labelEncoder = LabelEncoder()
scaler = MinMaxScaler()

data = pd.read_csv('data_w_genres.csv')
data.drop(['key', 'mode', 'count'], axis=1, inplace=True)
columns = ['instrumentalness', 'acousticness', 'danceability','energy','liveness','genres']

newData = data.replace(0, pd.np.nan).dropna(axis=0, how='any')
newData = newData[columns]

newData = newData[newData.genres != "[]"]
newData.reset_index(drop=True, inplace=True)
filterDataArray = []
rap = 0
classical = 0

for index, row in newData.iterrows():
    if("rap" in row['genres'] and rap <1100):
        newData.at[index, 'genres'] = 'rap'
        filterDataArray.append(newData.iloc[index])
        rap = rap + 1
    elif("classical" in row['genres'] and classical <1100):
        newData.at[index, 'genres'] = 'classical'
        filterDataArray.append(newData.iloc[index])
        classical = classical + 1
        
        
        
filterData = pd.DataFrame(filterDataArray, columns = ['instrumentalness', 'acousticness', 'danceability','energy','liveness', 'genres'])
labelEncoder = LabelEncoder()
labelEncoder.fit(filterData['genres'])
filterData['genres'] = labelEncoder.transform(filterData['genres'])


x = np.array(filterData.drop(['genres'], 1).astype(float))
filterData= filterData.drop(['genres'], 1)

X_scaled = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=2, max_iter=600).fit(X_scaled)


filterData['kmeans'] = kmeans.labels_
filterData.columns = ['instrumentalness', 'acousticness', 'danceability','energy','liveness','kmeans']

fig = px.scatter_3d(filterData, x='instrumentalness', y='acousticness', z='danceability',
              color='kmeans')
fig.show()

c0 = filterData[filterData['kmeans']==0]
c1 = filterData[filterData['kmeans']==1]


c0.drop(['kmeans'], axis=1, inplace=True)
c1.drop(['kmeans'], axis=1, inplace=True)

x = c0.values #returns a numpy array
c0_scaled = scaler.fit_transform(x)
c0 = pd.DataFrame(c0_scaled)
c0.columns = ['instrumentalness', 'acousticness', 'danceability','energy','liveness' ]
c0=c0.melt(var_name='groups', value_name='rap')

x = c1.values #returns a numpy array
c1_scaled = scaler.fit_transform(x)
c1 = pd.DataFrame(c1_scaled)
c1.columns = ['instrumentalness', 'acousticness', 'danceability','energy','liveness']
c1=c1.melt(var_name='groups', value_name='classical')


f, axes = plt.subplots(2, 1)
ax = sns.violinplot( data=c0 ,x="groups", y="rap", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="classical", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])

plt.show()