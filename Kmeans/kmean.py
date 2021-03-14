import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
kmeans = KMeans(n_clusters=10, max_iter=600, algorithm = 'auto')


data = pd.read_csv('data_by_year.csv')
pdx = pd.read_csv('data_by_year.csv')
pdy = pd.read_csv('data_by_year.csv')

y = np.array(data['year'])
X = np.array(data.drop(['year', 'key', 'mode'], 1))

for i in range(len(y)):
    if y[i] == 1930:
        y[i] = 0
    elif y[i] == 1940:
        y[i] = 1
    elif y[i] == 1950:
        y[i] = 2
    elif y[i] == 1960:
        y[i] = 3
    elif y[i] == 1970:
        y[i] = 4 
    elif y[i] == 1980:
        y[i] = 5
    elif y[i] == 1990:
        y[i] = 6
    elif y[i] == 2000:
        y[i] = 7
    elif y[i] == 2010:
        y[i] = 8
    elif y[i] == 2020:
        y[i] = 9

X_train, X_test, y_train, y_test = train_test_split(pdx, pdy, test_size=0.33, random_state=42)

print("***** Train_Set X*****")
print(X_train.head())
print("\n")


print("***** Train_Set Y*****")
print(y_train.head())
print("\n")

X_scaled = scaler.fit_transform(X)

kmeans.fit(X_scaled)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i])
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
       correct += 1

print(correct/len(X))

y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
