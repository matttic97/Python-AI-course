import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

svc = svm.SVC(kernel = 'linear', C = 2)
svc.fit(x_train, y_train)

predictions = svc.predict(x_test)
s_score = metrics.accuracy_score(y_test, predictions)

knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(x_train, y_train)

predictions = knn.predict(x_test)
k_score = metrics.accuracy_score(y_test, predictions)

print('SVC:', s_score, '\nKNN:', k_score)
