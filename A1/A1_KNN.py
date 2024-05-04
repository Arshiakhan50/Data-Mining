import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# confusion matrices
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns 
from IPython.display import display
# cross validation
from sklearn.model_selection import cross_val_score

data_2018 = pd.read_csv("2018.csv")
data_2019 = pd.read_csv("2019.csv")

data_2018['Score'] = data_2018['Score'].apply(lambda x: 1 if x >= 5.43 else 0)
data_2019['Score'] = data_2019['Score'].apply(lambda x: 1 if x >= 5.43 else 0)

X_train = data_2019[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_train = data_2019['Score']

X_2018 = data_2018[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_2018 = data_2018['Score']

X_train = scale(X_train)
X_2018 = scale(X_2018)

# Cross validation
k_values = [i for i in range (1,31)]
scores = []
# essentially pass in k values from 1-31, then check the best performing one
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=5)
    scores.append(np.mean(score))
    # cross_val_score = split into 5 consecute folds (cv = 5)
    # train on 4, test on 1, then return the accuracyof each fold on the test set
    
best_index = np.argmax(scores)
best_k = k_values[best_index]


knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
ypred = knn.predict(X_2018)
print("Accuracy:", round(accuracy_score(y_2018, ypred)*100), "%")
# Confusion matrix
cm = metrics.confusion_matrix(y_2018, ypred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Yes', 'No'])

cm_display.plot()
plt.title("KNN Confusion Matrix")
plt.show() 

# Attribute relevance
data = {'GDP per capita': [0], 'Social support': [0], 'Freedom to make life choices': [0], 'Generosity': [0]}
attribute_imp = pd.DataFrame(data)

# first column
test = X_2018.copy()
test[:, 0] = np.random.permutation(test[:, 0])
data['GDP per capita'] = 100*(accuracy_score(y_2018, ypred) - knn.score(test, y_2018))
# collect the significae of the second column
test1 = X_2018.copy()
test1[:, 1] = np.random.permutation(test1[:, 1])
data['Social Support'] = 100*(accuracy_score(y_2018, ypred) - knn.score(test1, y_2018))
# Third attribute
test2 = X_2018.copy()
test2[:, 2] = np.random.permutation(test2[:, 2])
data['Social Freedom to make life choices'] = 100*(accuracy_score(y_2018, ypred) - knn.score(test2, y_2018))
# fourth attribute
test3 = X_2018.copy()
test3[:, 3] = np.random.permutation(test3[:, 3])
data['Generosity'] = 100*(accuracy_score(y_2018, ypred) - knn.score(test3, y_2018))
# Generosity and freedom to make life choices have the least significatn impact 
# on happiness levels 

feature1 = 'GDP per capita'
feature2 = 'Social support'
new_data = data_2018.drop(['Overall rank', 'Country or region', 'Score', 'Perceptions of corruption', 'Healthy life expectancy'], axis=1)
# Get the indices of the selected features
feature1_index = new_data.columns.get_loc(feature1)
feature2_index = new_data.columns.get_loc(feature2)

# Extract the two features from the scaled training data
X_train_subset = X_train[:, [feature1_index, feature2_index]]

# Fit the KNN classifier on the subset of training data
knn.fit(X_train_subset, y_train)

# Create a meshgrid to plot the decision boundary
x_min, x_max = X_train_subset[:, 0].min() - 1, X_train_subset[:, 0].max() + 1
y_min, y_max = X_train_subset[:, 1].min() - 1, X_train_subset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict the class for each point in the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the training data
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_subset[:, 0], X_train_subset[:, 1], c=y_train, s=20, edgecolor='k')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title('KNN Decision Boundary')
plt.show()