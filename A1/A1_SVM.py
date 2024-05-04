import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
# confusion matrices
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns 
from IPython.display import display

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

SupportVectorClassModel = SVC()
SupportVectorClassModel.fit(X_train,y_train)
ypred = SupportVectorClassModel.predict(X_2018)

print("Accuracy:", round(accuracy_score(y_2018, ypred)*100), "%")
# Confusion matrix
cm = metrics.confusion_matrix(y_2018, ypred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Yes', 'No'])

cm_display.plot()
plt.title("SVM Confusion Matrix")
plt.show() 

perm_importance = permutation_importance(SupportVectorClassModel, X_2018, y_2018)

feature_names = ['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']  # Assuming the order of features
features = np.array(feature_names)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]
plt.bar(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.xticks(rotation=45)
plt.show()
