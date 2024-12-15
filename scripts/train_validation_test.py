from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import pandas as pd

csv_path = "Fe30.csv"
data = pd.read_csv(csv_path)

x = data.iloc[:, 0:11]
y = data.iloc[:, 11]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

x_train_, x_test_, y_train_, y_test_ = train_test_split(x_train,
                                                        y_train,
                                                        test_size=0.1,
                                                        random_state=0)



RF = RandomForestRegressor()

RF.fit(x_train_, y_train_)
y_pred = RF.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scores = cross_val_score(RF, x_train, y_train, cv=5)

print("RF cross-validation score:%.6f" % scores.mean())
print('RF R2:%.6f' % r2)
print('RF Mean Absolute Error:%.6f' % mae)
print('RF Mean Squared Error:%.6f' % mse)
print('RF Root Mean Squared Error:%.6f' % rmse)


GBR = GradientBoostingRegressor()

GBR.fit(x_train_, y_train_)
y_pred = GBR.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scores = cross_val_score(GBR, x_train, y_train, cv=5)

print("GB cross-validation score:%.6f" % scores.mean())
print('GB R2:%.6f' % r2)
print('GB Mean Absolute Error:%.6f' % mae)
print('GB Mean Squared Error:%.6f' % mse)
print('GB Root Mean Squared Error:%.6f' % rmse)


tree = DecisionTreeRegressor()
tree.fit(x_train_, y_train_)
y_pred = tree.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scores = cross_val_score(tree, x_train, y_train, cv=5)

print("DR cross-validation score:%.6f" % scores.mean())
print('DR R2:%.6f' % r2)
print('DR Mean Absolute Error:%.6f' % mae)
print('DR Mean Squared Error:%.6f' % mse)
print('DR Root Mean Squared Error:%.6f' % rmse)


NN = MLPRegressor()

NN.fit(x_train_, y_train_)
y_pred = NN.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scores = cross_val_score(NN, x_train, y_train, cv=5)

print("NN cross-validation score:%.6f" % scores.mean())
print('NN R2:%.6f' % r2)
print('NN Mean Absolute Error:%.6f' % mae)
print('NN Mean Squared Error:%.6f' % mse)
print('NN Root Mean Squared Error:%.6f' % rmse)

svr = SVR()

svr.fit(x_train_, y_train_)
y_pred = svr.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scores = cross_val_score(svr, x_train, y_train, cv=5)

print("SVR cross-validation score:%.6f" % scores.mean())
print('SVR R2:%.6f' % r2)
print('SVR Mean Absolute Error:%.6f' % mae)
print('SVR Mean Squared Error:%.6f' % mse)
print('SVR Root Mean Squared Error:%.6f' % rmse)


lr = LinearRegression()
lr.fit(x_train_, y_train_)
y_pred = lr.predict(x_test)

r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
scores = cross_val_score(lr, x_train, y_train, cv=5)

print("Linear cross-validation score:%.6f" % scores.mean())
print('Linear R2:%.6f' % r2)
print('Linear Mean Absolute Error:%.6f' % mae)
print('Linear Mean Squared Error:%.6f' % mse)
print('Linear Root Mean Squared Error:%.6f' % rmse)
