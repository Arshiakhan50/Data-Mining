import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Testing and Training Data
trainingData = pd.read_csv("2019.csv")
testingData = pd.read_csv("2018.csv")

trainingData['Score'] = trainingData['Score'].apply(lambda x: 1 if x >= 5.43 else 0)
testingData['Score'] = testingData['Score'].apply(lambda x: 1 if x >= 5.43 else 0)

x_train = trainingData[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_train = trainingData['Score']

x_test = testingData[['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity']]
y_test = testingData['Score']

# Scale the data for preprocessing
x_train = scale(x_train)
x_test = scale(x_test)

# Initialize and train the ANN classifier
clf = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='lbfgs', max_iter=400, random_state=40)
clf.fit(x_train, y_train)

# Predict the labels for test data
y_pred = clf.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy*100), "%")

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Yes', 'No'])
disp.plot()
plt.title("ANN Confusion Matrix")
plt.show()


# Extract weights connecting input features to the first hidden layer
weights_input_hidden = clf.coefs_[0]

# Calculate feature importances as the sum of absolute weights for each feature
feature_importances = np.abs(weights_input_hidden).sum(axis=1)

# Normalize feature importances
feature_importances /= feature_importances.sum()

# Plot feature importances
plt.bar(range(len(feature_importances)), feature_importances, tick_label=['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity'])
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance")
# Tilt x-axis labels
plt.xticks(ticks=np.arange(len(feature_importances)), labels=['GDP per capita', 'Social support', 'Freedom to make life choices', 'Generosity'], rotation=30)

plt.show()