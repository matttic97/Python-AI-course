import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plot
from matplotlib import style
import pickle

data = pd.read_csv('data/student/student-mat.csv', sep =';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]

predict_label = 'G3'

X = np.array(data.drop([predict_label], 1))
Y = np.array(data[predict_label])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

'''best_score = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    model = linear_model.LinearRegression()

    model.fit(x_train, y_train)
    model_score = model.score(x_test, y_test)
    print(model_score)

    # save model
    if model_score > best_score:
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(model, f)'''

# load model
model_to_load = open('studentmodel.pickle', 'rb')
model = pickle.load(model_to_load)

print('Coefficient  \n', model.coef_)
print('Intercept \n', model.intercept_)

# prediction
predictions = model.predict(x_test)
print(predictions.astype(int) - y_test)


style.use('ggplot')
p = 'G2'
plot.scatter(data[p], data[predict_label])
plot.xlabel(p)
plot.ylabel('Final grade')
plot.show()
