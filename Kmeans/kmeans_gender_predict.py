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

data = pd.read_csv('data_w_genres.csv')

data.drop(['key', 'mode', 'count'], axis=1, inplace=True)
newData = data[['instrumentalness', 'acousticness', 'danceability', 'genres']]

newData = newData[newData.genres != "[]"] # Removing data without gender
newData.reset_index(drop=True, inplace=True) # Reindexing 

# Generate new DF only with the genders we want
filterDataArray = []
rock = 0
rap = 0
jazz = 0
classical = 0

# Getting 1100 data from rock rap jazz and classical
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
        

# Replacing gender string with integers
filterData = pd.DataFrame(filterDataArray, columns = ['instrumentalness', 'acousticness', 'danceability', 'genres'])
labelEncoder = LabelEncoder()
labelEncoder.fit(filterData['genres'])
filterData['genres'] = labelEncoder.transform(filterData['genres'])

# Spliting data to create a Training and testing data
filterDataOutput = np.array(filterData['genres'])
filterDataInputs = np.array(filterData.drop(columns=['genres']))

X_train, X_test, y_train, y_test = train_test_split(filterDataInputs, filterDataOutput, test_size=0.33, random_state=42)


# Kmeans
X_scaled = scaler.fit_transform(X_train)
kmeans = KMeans(n_clusters=4, max_iter=10000).fit(X_scaled)

# Predicting  
correct = 0
y_kmeans = kmeans.predict(X_test)

# Checking the accuracy of prediction
for i in range(len(y_test)):
    if y_test[i] == y_kmeans[i]:
       correct += 1

print(correct/len(y_kmeans))
