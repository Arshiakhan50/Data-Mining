import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
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

# Gaussian Naive Bayes model
nb = GaussianNB()

# fit the model
nb.fit(X_train, y_train)
ypred = nb.predict(X_2018)
print("Accuracy:", round(accuracy_score(y_2018, ypred)*100), "%")


# print accuracy for Gaussian Naive Bayes
imps = permutation_importance(nb, X_2018, y_2018)
print(imps.importances_mean)

# Get the mean importances and their corresponding indices
feature_names = ['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']
mean_importances = imps.importances_mean
indices = np.argsort(mean_importances)[::-1]  # Sort in descending order

# Print the two most important features
print("Most important features:")
for i in range(2):
    feature_index = indices[i]
    feature_importance = mean_importances[feature_index]
    feature_name = feature_names[feature_index]
    print(f"{feature_name}: {feature_importance}")

cm = metrics.confusion_matrix(y_2018, ypred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Yes', 'No'])


cm_display.plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show() 

