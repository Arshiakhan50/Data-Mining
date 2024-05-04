import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

trainingData = pd.read_csv("2019.csv")
testingData = pd.read_csv("2018.csv")

trainingData['Score'] = trainingData['Score'].apply(lambda x: 1 if x >= 5.43 else 0)
testingData['Score'] = testingData['Score'].apply(lambda x: 1 if x >= 5.43 else 0)
# DECISION TREE

x_train = trainingData[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_train = trainingData['Score']

x_test = testingData[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_test = testingData['Score']

# Create a decision tree classifier object
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Train the classifier
classifier.fit(x_train, y_train)

# Get feature importances
importances = classifier.feature_importances_

# Print feature importances
print("Feature Importances:")
for feature, importance in zip(x_train.columns, importances):
    print(f"{feature}: {importance}")


# Prepare the test trainingData
testData = testingData.drop(['Overall rank', 'Country or region', 'Perceptions of corruption', 'Healthy life expectancy'], axis=1)
x_test =  testData[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_test = testData['Score']


# Apply the decision tree to classify the test records
predictions = classifier.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy*100), "%")

# Plot the decision tree
plt.figure(figsize=(12, 8))  # Adjust the figsize as needed
plot_tree(classifier, feature_names=x_train.columns, class_names=['0', '1'], filled=True, fontsize=10)
plt.title("Decision Tree ")
plt.show()

# Compute confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Yes', 'No'])


cm_display.plot()
plt.title("Decision Tree Confusion Matrix")

# Plot feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x=x_train.columns, y=importances )
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()