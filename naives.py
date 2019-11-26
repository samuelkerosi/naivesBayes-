import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


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


#  ouput file stuff
outfile = open("Naives_outfile.txt", "w")
outfile.write("Model Accuracy: \n")
outfile.write(str(accuracy))
outfile.write("\n")
outfile.write("\n")
outfile.write("Predicted Class Labels \n")
outfile.write(str(prediction))
outfile.write("\n")
outfile.write("\n")
outfile.write("Test Class Labels: \n")
outfile.write(str(y_test))
outfile.close()
