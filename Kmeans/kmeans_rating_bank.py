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

data = pd.read_csv('data_bank.csv')
data.drop(['RowNumber', 'CustomerId', 'Surname','Geography','Gender'], axis=1, inplace=True)
newData = data.dropna() # Removing data without gender



filterDataOutput = np.array(newData['Exited'].astype(float))
filterDataInputs = np.array(newData.drop(['Exited'],1).astype(float))


X_train, X_test, y_train, y_test = train_test_split(filterDataInputs, filterDataOutput, test_size=0.3, random_state=42)


X_scaled = scaler.fit_transform(X_train)
kmeans = KMeans(n_clusters=2, max_iter=300).fit(X_scaled)

correct = 0
y_kmeans = kmeans.predict(X_test)

for i in range(len(y_test)):
    if y_test[i] == y_kmeans[i]:
       correct += 1

print(correct/len(y_kmeans))

'''
newData = newData.drop(['stroke'], 1)

c0 = newData[newData['kmeans']==0]
c1 = newData[newData['kmeans']==1]


c0.drop(['kmeans'], axis=1, inplace=True)
c1.drop(['kmeans'], axis=1, inplace=True)


x = c0.values #returns a numpy array
c0_scaled = scaler.fit_transform(x)
c0 = pd.DataFrame(c0_scaled)
c0.columns = newData.columns.drop(['kmeans'])
c0=c0.melt(var_name='groups', value_name='Cluster1')

x = c1.values #returns a numpy array
c1_scaled = scaler.fit_transform(x)
c1 = pd.DataFrame(c1_scaled)
c1.columns =newData.columns.drop(['kmeans'])
c1=c1.melt(var_name='groups', value_name='Cluster2')


f, axes = plt.subplots(2, 1)
ax = sns.violinplot( data=c0 ,x="groups", y="Cluster1", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[0])
ax = sns.violinplot( data=c1 ,x="groups", y="Cluster2", linewidth = 0.6, inner = 'point', scale= 'width', ax=axes[1])

plt.show()'''