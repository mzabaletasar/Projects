import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
from kneed import KneeLocator


def assign_cluster(row): 
    return np.argmin([np.sqrt(np.sum(arr)) for arr in (row - np.array(centroids))**2])+1
    
def convg_criteria(dataframe):
    num_cambios = (df['cluster'] - list(map(assign_cluster, 
                                df.loc[:,df.columns!='cluster'].values))).astype(bool).sum(axis=0)
    return num_cambios < (0.05 * df.shape[0])
    
def recalculate_centroids(k):
    if len(df.loc[df['cluster']==k, df.columns[:-1]]) == 0:
        return centroids[k-1]
    else:
        return np.mean(df.loc[df['cluster']==k, df.columns[:-1]], axis=0)
    
def calculate_sse(k):
    return np.sum(np.sum(( df.loc[df['cluster']==k,df.columns[:-1]] - centroids[k-1] )**2))


st = time.time()

df = pd.read_csv('C:/Máster/1.1/Fundamentals/Week 3-4/lab2/computers.csv')
df['cd'] = pd.get_dummies(df['cd']).drop('no', axis=1)
df['laptop'] = pd.get_dummies(df['laptop']).drop('no', axis=1)
df.pop('id')

data = df.values
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = min_max_scaler.fit_transform(data)
df = pd.DataFrame(data_scaled, columns=df.columns)


sse_points = []
for num_clusters in range(1,11):
    print('vuelta',num_clusters)
    centroids = []
    for i in range(num_clusters):
        centroids.append(np.random.random(df.shape[1]))
    
    convg_result = False
    i = 1
    while convg_result == False:
        print('iteracion', i)
        print('empieza assign cluster')
        cluster_assignment = list(map(assign_cluster, df.loc[:, df.columns!='cluster'].values))
        print('cluster asignado')
        df['cluster'] = cluster_assignment
        print('recalculando centroids')
        centroids = list(map(recalculate_centroids, range(1,num_clusters+1)))
        print('centroids calculados, empieza convg criteria')
        convg_result = convg_criteria(df)
        print('convg criteria hecho')
        i = i+1
    sse_points.append(sum(map(calculate_sse, range(1,num_clusters+1))))
    df.pop('cluster')
    

x = range(1,len(sse_points)+1)
kn = KneeLocator(x, sse_points, curve='convex', direction='decreasing')
print('The optimal number of clusters (k) is:', kn.knee)

print('Clustering now with optimal k ...')

num_clusters = kn.knee
centroids = []
for i in range(num_clusters):
    centroids.append(np.random.random(df.shape[1]))

convg_result = False
i = 1
while convg_result == False:
    cluster_assignment = list(map(assign_cluster, df.loc[:, df.columns!='cluster'].values))
    df['cluster'] = cluster_assignment
    centroids = list(map(recalculate_centroids, range(1,num_clusters+1)))
    convg_result = convg_criteria(df)
    i = i+1
    
et = time.time()
print('Total execution time:', et-st)


# Plot first 2 dimensions of clusters
for k in range(1,num_clusters+1):
    plt.scatter(df.loc[df['cluster']==k,'price'], df.loc[df['cluster']==k,'speed'], label = k,s=3)
plt.legend()
plt.show()

# Cluster with highest average price:
ma=np.max(data[:,0])
mi=np.min(data[:,0])
print('Cluster', df.groupby('cluster')['price'].mean().idxmax(),
      'has the highest average price of:', df.groupby('cluster')['price'].mean().max()*(ma-mi)+mi)

# Plot heat map
fig, ax = plt.subplots()
im = ax.imshow(centroids, cmap='hot_r', interpolation = 'nearest')
ax.set_xticks(np.arange(len(df.columns[:-1])))
ax.set_yticks(np.arange(len(range(1,num_clusters+1))))
    
ax.set_xticklabels(df.columns[:-1])
ax.set_yticklabels(range(1,num_clusters+1))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

ax.set_title('Centroids Heatmap')
fig.tight_layout()
cbar=plt.colorbar(im)
cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['lowest\n value','','','','','','','','highest\n value']):
    cbar.ax.text(1.2, (0.75 * j + 0.3) / 8.0, lab)
cbar.ax.get_yaxis().labelpad = 15

# Plot of Elbow graph:
plt.figure(figsize=(16,8))
plt.plot(x, sse_points, 'bx-')
plt.xlabel('num of clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow graph')
plt.show()

