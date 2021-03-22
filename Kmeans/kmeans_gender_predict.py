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

newData = newData[newData.genres != "[]"] # Removing data without gender
newData.reset_index(drop=True, inplace=True) # Reindexing 

# Generate new DF only with the genders we want
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
        

# Replacing gender string with integers
filterData = pd.DataFrame(filterDataArray, columns = columns)
labelEncoder = LabelEncoder()
labelEncoder.fit(filterData['genres'])
filterData['genres'] = labelEncoder.transform(filterData['genres'])

# Spliting data to create a Training and testing data
filterDataOutput = np.array(filterData['genres'])
filterDataInputs = np.array(filterData.drop(['genres'],1).astype(float))
X_train, X_test, y_train, y_test = train_test_split(filterDataInputs, filterDataOutput, test_size=0.2, random_state=42)


# Kmeans
X_scaled = scaler.fit_transform(X_train)
kmeans = kmeans = KMeans(n_clusters=2, max_iter=600)
kmeans.fit(X_scaled)

# Predicting  
correct = 0

# Checking the accuracy of prediction
for i in range(len(X_test)):
    predict_me = np.array(X_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y_test[i]:
        correct += 1

print("Acerto " + str((100*correct)/len(X_test)))

correct = 0
y_test = [0]*414
for i in range(len(X_test)):
    predict_me = np.array(X_test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y_test[i]:
        correct += 1

print("Chutando " + str((100*correct)/len(X_test)))
