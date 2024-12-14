import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import metrics

csv_path = "Fe30.csv"

data = pd.read_csv(csv_path)
x = data.iloc[:, 0:9] ### Not considering the importance of the adsorbate itself. (ads and ads(mass))
y = data.iloc[:, 11]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regressor = RandomForestRegressor(max_features=1,
                                  n_jobs=-1
                                  )

regressor.fit(x_train, y_train)

importances = list(regressor.feature_importances_)
feature_list = list(data.columns)[0:9]
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
x_values = list(range(len(importances)))

plt.figure(figsize=(11, 7.0))
plt.bar(x_values, importances, orientation='vertical')
plt.xticks(x_values, feature_list)
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 11})
plt.yticks(fontproperties='Times New Roman', size=11)
plt.xticks(fontproperties='Times New Roman', size=11)
plt.savefig("feature.png", dpi=300)

