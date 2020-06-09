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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

lm = LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_pred))

lm = LinearRegression()
rfe_15 = RFE(lm, 15)
rfe_15.fit(X_train, y_train)
print(rfe_15.support_)
print(rfe_15.ranking_)
y_pred = rfe_15.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_pred))

lm = LinearRegression()
rfe_6 = RFE(lm, 6)
rfe_6.fit(X_train, y_train)
print(rfe_6.support_)
print(rfe_6.ranking_)
y_pred = rfe_6.predict(X_test)
print(r2_score(y_true=y_test, y_pred=y_pred))

col_15 = X_train.columns[rfe_15.support_]
X_train_rfe_15 = X_train[col_15]
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
lm_15 = sm.OLS(y_train, X_train_rfe_15).fit()
print(lm_15.summary())

X_test_rfe_15 = X_test[col_15]
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
X_test_rfe_15.info()
y_pred = lm_15.predict(X_test_rfe_15)
print(r2_score(y_true=y_test, y_pred=y_pred))

col_6 = X_train.columns[rfe_6.support_]
X_train_rfe_6 = X_train[col_6]
X_train_rfe_6 = sm.add_constant(X_train_rfe_6)
lm_6 = sm.OLS(y_train, X_train_rfe_6).fit()
print(lm_6.summary())

X_test_rfe_6 = X_test[col_6]
X_test_rfe_6 = sm.add_constant(X_test_rfe_6, has_constant='add')
X_test_rfe_6.info()
y_pred = lm_6.predict(X_test_rfe_6)
print(r2_score(y_true=y_test, y_pred=y_pred))

n_features_list = list(range(4, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(4, 20):
    lm = LinearRegression()
    rfe_n = RFE(lm, n_features)
    rfe_n.fit(X_train, y_train)
    col_n = X_train.columns[rfe_n.support_]
    X_train_rfe_n = X_train[col_n]
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)
    lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    adjusted_r2.append(lm_n.rsquared_adj)
    r2.append(lm_n.rsquared)
    X_test_rfe_n = X_test[col_n]
    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')
    y_pred = lm_n.predict(X_test_rfe_n)
    test_r2.append(r2_score(y_test, y_pred))
    

plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label='adjusted_r2')
plt.plot(n_features_list, r2, label='train_r2')
plt.plot(n_features_list, test_r2, label='test_r2')
plt.legend(loc='upper left')
plt.show()

