import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

# classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier

# show all of df
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# open csv with data
df = pd.read_csv('model2.csv')
df_c = df.copy()
del df_c['target'] 

X, y = df_c.values, df['target']

# normalize
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)

# divide data for train and test 
r_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=r_state)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=r_state), 
    'SGD huber': SGDClassifier(loss='huber', random_state=r_state),
    'Ridge cholesky': RidgeClassifier(solver='cholesky'),
    'KNN': KNeighborsClassifier(n_neighbors=142),
    'Gaussian NB': GaussianNB(), 
    'Multinomial NB': MultinomialNB(),
    'Random Forest': RFC(random_state=r_state),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=r_state)
}

# train each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    c_matrix = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (col for l in c_matrix for col in l)
    print('-' * 50)
    print(model_name)
    print('Matriz de confusao: ', c_matrix)
    print('accuracy: ', (tp+tn)/(tp+tn+fp+fn))
    print('precision: ', tp/(tp+fp))
    print('recall: ', tp/(tp+fn))
    print('score f1: ', 2 * (tp/(tp+fp) * tp/(tp+fn)) / (tp/(tp+fp) + tp/(tp+fn)))
    print('-' * 50, end='\n\n')
