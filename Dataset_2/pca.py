import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
import statsmodels.api as sm

cars = pd.read_csv('CarPrice_Assignment.csv')
cars['CarName'] = cars['CarName'].apply(lambda x: x.split()[0])
value_cars = cars.select_dtypes(include=['float64', 'int64'])
value_cars = value_cars.drop(['car_ID', 'symboling'], axis=1)

cor = value_cars.corr()
plt.figure(figsize=(16, 8))
sns.heatmap(cor, cmap='YlGnBu', annot=True)
plt.show()

cars.loc[(cars['CarName'] == 'vw') | (cars['CarName'] == 'vokswagen'), 'CarName'] = 'volkswagen'
cars.loc[cars['CarName'] == 'toyouta', 'CarName'] = 'toyota'
cars.loc[cars['CarName'] == 'porcshce', 'CarName'] = 'porsche'
cars.loc[cars['CarName'] == 'Nissan', 'CarName'] = 'nissan'
cars.loc[cars['CarName'] == 'maxda', 'CarName'] = 'mazda'
corr_matrix = cars.corr().abs()

cars = cars.drop(['car_ID', 'symboling'], axis=1)
X = cars.loc[:, cars.columns != 'price']
y = cars['price']
cars_categorical = cars.select_dtypes(include=['object'])
X = X.drop(list(cars_categorical.columns), axis=1)
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
X = pd.concat([X, cars_dummies], axis=1)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_ 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_pred))

X1 = sm.add_constant(X_train)
result = sm.OLS(y_train, X1).fit()
print (result.rsquared, result.rsquared_adj)
X_test = sm.add_constant(X_test)
y_pred = result.predict(X_test)
print (r2_score(y_test, y_pred))

n_features_list = list(range(4, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):
    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, train_size=0.7, test_size=0.3, random_state=100)
    lm_n = LinearRegression()
    lm_n.fit(X_train, y_train)
    X1 = sm.add_constant(X_train)
    lm_n = sm.OLS(y_train, X1).fit()
    adjusted_r2.append(lm_n.rsquared_adj)
    r2.append(lm_n.rsquared)
    X_test = sm.add_constant(X_test)
    y_pred = lm_n.predict(X_test)
    test_r2.append(r2_score(y_test, y_pred))
    

plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label='adjusted_r2')
plt.plot(n_features_list, r2, label='train_r2')
plt.plot(n_features_list, test_r2, label='test_r2')
plt.legend(loc='upper left')
plt.show()

