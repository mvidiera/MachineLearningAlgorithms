# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):  #approx, no of cluster from 1 to 10 (11 is ignored)
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #dist btw dp n centr, inertia is sum of dist and append in wcss[]
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# in graph I got to know that there is a slope n dispersion btw 4 n 6, so we consider as 5

# Fitting K-Means to the dataset. We got k as 5 so Im applying than taking range btw 1 to 10
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# we hve plotted graph and got DS. We still dont know which color means wt data. its a black box. I dono cluster 1's data
# now I need to observe that what data is clustered into clus1, 2 ... 
# We got so many clusters. Eacg DP in the cluster has similartity thats y its in same cluster.
# now we have new DP and I need to put that into single cluster. 
#I will check the properties of that DP and check the property(manual observation). In this ex, property is DISTANCE 
# 5 category- low income-low expenditure, low n- high ex, avg in- avg ex, high in- low ex, high in- high ex

# ex: annual income is 21, what will be score? there can be infinite expenditure. we cant get an exact value through graph.
# we have 5 cluster. group the data. We need to apply Regression model over clustering. Preferably Linear regression
# suppose income is 21, there are 2 cluster for 21. green n cyan. so take each cluster separately and apply linear regression
# we get 2 predicted values. One with green with least possibilties and 

#outliers- we generally find difficulties in outliers. Here for cyan, blue is outlier and viceversa. so we have taken each cluster
#separately and run regression over it. So there is no outliers in cluster, but we can label group as outliers.