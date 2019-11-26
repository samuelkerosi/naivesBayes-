import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score

trainingFile = np.genfromtxt("irisPCTraining.txt")
X_train = trainingFile[:, :-1]
y_train = trainingFile[:, -1]
testingFile = np.genfromtxt("irisPCTesting.txt")
X_test = testingFile[:, :-1]
y_test = testingFile[:, -1]
fit_model = GaussianNB()
fit_model.fit(X_train, y_train)
prediction = fit_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

# confustion matrices
cmatrix = confusion_matrix(y_test, prediction)
print("Matrix \n", cmatrix)
true_positives = cmatrix[0][0]
false_positives = cmatrix[1][0]
true_negatives = cmatrix[1][1]
false_negatives = cmatrix[0][1]

# precision and recalling
precsion = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)

#  ouput file stuff
outfile = open("Naives_outfile.txt", "w")
outfile.write("Model Accuracy: \n")
outfile.write(str(accuracy))
outfile.write("\n")
outfile.write("\n")
outfile.write("Predicted Class Labels (irisPC datasets used here) \n")
outfile.write(str(prediction))
outfile.write("\n")
outfile.write("\n")
outfile.write("True Positives: \n")
outfile.write(str(true_positives))
outfile.write("\n")
outfile.write("False Positives: \n")
outfile.write(str(false_positives))
outfile.write("\n")
outfile.write("True Negatives: \n")
outfile.write(str(true_negatives))
outfile.write("\n")
outfile.write("False Negatives: \n")
outfile.write(str(false_negatives))
outfile.write("\n")
outfile.write("\n")
outfile.write("Precision: \n")
outfile.write(str(precsion))
outfile.write("\n")
outfile.write("\n")
outfile.write("Recall: \n")
outfile.write(str(recall))


outfile.close()
