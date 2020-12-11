import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB

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

# divide train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000), 
    'Gaussian NB': GaussianNB(), 
    'Multinomial NB': MultinomialNB(),
    'Complement NB': ComplementNB(),
    'BernoulliNB': BernoulliNB(),
    'KNN': KNeighborsClassifier(n_neighbors=142)
}

# train each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_score = model.score(X_test, y_test)
    print(model_name)
    print('Score: ', model_score)
    print('Matriz de confusao: ', metrics.confusion_matrix(y_test, y_pred))
