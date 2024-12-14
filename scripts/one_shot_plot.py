from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from Analysis import analysis as ana
import matplotlib.pyplot as plt

csv_path = "Fe30.csv"

data = ana().dataframe(csv_path)
x = data.iloc[:, 0:11]
y = data.iloc[:, 11]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

regressor = RandomForestRegressor()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))


fig = plt.figure(figsize=(7.0, 7.0))
ax = fig.add_subplot(111)

eb_min = min(y_test)
eb_max = max(y_test)

plt.plot(y_test, y_pred, "o", markersize=3)
plt.plot([eb_min, eb_max], [eb_min, eb_max], "r-", lw=0.3)
plt.xlabel("Eb$_{true}$ (eV)", fontdict={"weight": "heavy", "size": 16})
plt.ylabel("Eb$_{pred}$ (eV)", fontdict={"weight": "heavy", "size": 16})
plt.title("Binding Energies", fontdict={"weight": "heavy", "size": 16})

ax.text(0.95, 0.5, "MAE: %.4f eV\nRMSE: %.4f eV" % (mae, rmse), size=12,
         horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 16})
plt.yticks(fontproperties='Times New Roman', size=16, weight="heavy")
plt.xticks(fontproperties='Times New Roman', size=16, weight="heavy")
plt.savefig("Pred_on_testing.png", dpi=300)
