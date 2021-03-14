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
newData = data[['instrumentalness', 'acousticness', 'danceability', 'genres']]

newData = newData[newData.genres != "[]"]
newData.reset_index(drop=True, inplace=True)
filterDataArray = []
rock = 0
rap = 0
jazz = 0
classical = 0

for index, row in newData.iterrows():
    if("rock" in row['genres'] and rock <1100):
        newData.at[index, 'genres'] = 'rock'
        filterDataArray.append(newData.iloc[index])
        rock = rock + 1
    elif("rap" in row['genres'] and rap <1100):
        newData.at[index, 'genres'] = 'rap'
        filterDataArray.append(newData.iloc[index])
        rap = rap+1
    elif("jazz" in row['genres'] and jazz <1100):
        newData.at[index, 'genres'] = 'jazz'
        filterDataArray.append(newData.iloc[index])
        jazz = jazz + 1
    elif("classical" in row['genres'] and classical <1100):
        newData.at[index, 'genres'] = 'classical'
        filterDataArray.append(newData.iloc[index])
        classical = classical + 1
        
        
        
print(rock,rap,jazz,classical)
filterData = pd.DataFrame(filterDataArray, columns = ['instrumentalness', 'acousticness', 'danceability', 'genres'])
labelEncoder = LabelEncoder()
labelEncoder.fit(filterData['genres'])
filterData['genres'] = labelEncoder.transform(filterData['genres'])


x = np.array(filterData.drop(['genres'], 1).astype(float))
filterData= filterData.drop(['genres'], 1)

X_scaled = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=4, max_iter=600).fit(X_scaled)


filterData['kmeans'] = kmeans.labels_
filterData.columns = ['instrumentalness', 'acousticness', 'danceability','kmeans']

fig = px.scatter_3d(filterData, x='instrumentalness', y='acousticness', z='danceability',
              color='kmeans')
fig.show()

c0 = filterData[filterData['kmeans']==0]
c1 = filterData[filterData['kmeans']==1]
c2 = filterData[filterData['kmeans']==2]
c3 = filterData[filterData['kmeans']==3]


c0.drop(['kmeans'], axis=1, inplace=True)
c1.drop(['kmeans'], axis=1, inplace=True)
c2.drop(['kmeans'], axis=1, inplace=True)
c3.drop(['kmeans'], axis=1, inplace=True)

x = c0.values #returns a numpy array
c0_scaled = scaler.fit_transform(x)
c0 = pd.DataFrame(c0_scaled)
c0.columns = ['instrumentalness', 'acousticness', 'danceability' ]
c0=c0.melt(var_name='groups', value_name='rock')

x = c1.values #returns a numpy array
c1_scaled = scaler.fit_transform(x)
c1 = pd.DataFrame(c1_scaled)
c1.columns = ['instrumentalness', 'acousticness', 'danceability' ]
c1=c1.melt(var_name='groups', value_name='rap')

x = c2.values #returns a numpy array
c2_scaled = scaler.fit_transform(x)
c2 = pd.DataFrame(c2_scaled)
c2.columns = ['instrumentalness', 'acousticness', 'danceability']
c2=c2.melt(var_name='groups', value_name='jazz')

x = c3.values #returns a numpy array
c3_scaled = scaler.fit_transform(x)
c3 = pd.DataFrame(c3_scaled)
c3.columns = ['instrumentalness', 'acousticness', 'danceability']
c3=c3.melt(var_name='groups', value_name='classical')


f, axes = plt.subplots(4, 1)
ax = sns.violinplot( data=c0 ,x="groups", y="rock", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="rap", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])
ax = sns.violinplot( data=c2 ,x="groups", y="jazz", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[2])
ax = sns.violinplot( data=c3 ,x="groups", y="classical", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[3])

plt.show()