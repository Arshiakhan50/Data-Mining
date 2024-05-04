import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from apyori import apriori
from sklearn.preprocessing import StandardScaler


dataClustering = pd.read_csv("./MLS.csv")
dataClustering = dataClustering.iloc[1:2277]
dataClustering = dataClustering.fillna(np.nan)
dataClustering.drop(columns=["Date"], inplace=True)
dataClustering.dropna(inplace=True)
dataClustering = dataClustering.reset_index(drop=True)

# storing dataset classe
dataClusteringClass = dataClustering["Location"]

# storing the list of possible classes in the dataset
datasetClusteringClasses = dataClustering["Location"].unique()

# separating attributes from classes
dataClusteringAttributes = dataClustering.drop(columns=["Location"])
scaler = StandardScaler()
dataClusteringAttributes_standardized = scaler.fit_transform(dataClusteringAttributes)

# dictionary to store Sum of Squared Errors (SSE) for different values of k
sses = {}

# testing a range of k cluster numbers while computing their SSE scores for the elbow method
for k in range(1, dataClusteringClass.nunique()+1):
    kmeans = KMeans(n_clusters=k, n_init=100).fit(dataClusteringAttributes)
    sses[k] = kmeans.inertia_

# plotting SSE to cluster number graph for the elbow method
plt.figure()
plt.plot(list(sses.keys()), list(sses.values()))
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Sum of Squared Errors (SSE) vs Number of Clusters (k) [1e14 = 1 followed by 14 zeros]")
plt.show()

# according to the plotted SSE graph, we concluded the elbow to be at k=6
np.random.seed(42)  # Set a random seed for reproducibility
kmeans = KMeans(n_clusters=6, n_init=100).fit(dataClusteringAttributes)

# extracting cluster labels provided by the KMeans algorithm
clustersLabels = pd.DataFrame(kmeans.labels_, columns=['Cluster ID']) 

# concatenating the cluster labels to the original dataset
dataClustering = pd.concat([dataClustering, clustersLabels], axis=1)

# looping through each cluster label
for label in range(6):
    # extract all the rows that were labeled this label by the KMeans algorithm
    cluster = dataClustering.loc[dataClustering["Cluster ID"] == label]
    # count the number of different elements of various classes present in the resulting cluster
    classesInCluster = cluster["Location"].value_counts()
    # extracting the most common class in cluster
    mostCommonClassInCluster = cluster["Location"].mode()[0]
    # calculating purity of the cluster
    purity = classesInCluster[mostCommonClassInCluster] / len(cluster.index)
    # building value array for the bar graphs of the number of rows of each class in the cluster
    classCounts = []
    for datasetClusteringClass in datasetClusteringClasses:
        if datasetClusteringClass in classesInCluster:
            classCounts.append(classesInCluster[datasetClusteringClass])
        else:
            classCounts.append(0)
    plt.bar(x=datasetClusteringClasses, height=classCounts)
    plt.xticks(rotation=90)
    plt.xlabel("Location")  
    plt.ylabel("Frequency") 
    plt.title(f"Cluster {label}")  
    plt.tight_layout()  
    plt.show()
    print(f"Cluster ID: {label}, Most Common Class In Cluster: {mostCommonClassInCluster}, Purity: {purity}")

data = pd.read_csv("groceries - groceries.csv")
data = data.drop(columns=['Item(s)'])
transactions = []
for i in range(data.shape[0]):
    transactions.append(data.iloc[i].dropna().tolist())
    
# association analysis using the apriori algo
itemsets = apriori(transactions, min_support=0.02, min_confidence = 0.45)

# # left hand side of the arrow: items_base
print("Rules:")
for rule in itemsets:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        '\nSupport:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence, 'Lift:', rule.ordered_statistics[0].lift, "\n")