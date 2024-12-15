from sklearn.model_selection import train_test_split, LearningCurveDisplay, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


csv_path = "Fe30.csv"
data = pd.read_csv(csv_path)

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

model = RandomForestRegressor(n_jobs=-1)

common_params = {
    "X": x_train, # or x
    "y": y_train, # or y
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": -1,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 7), sharey=True)
LearningCurveDisplay.from_estimator(model, **common_params, ax=ax)
handles, label = ax.get_legend_handles_labels()
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})

plt.ylabel("Accuracy", fontdict={"weight": "heavy", "size": 14})
plt.xlabel("Number of samples in the training set", fontdict={"weight": "heavy", "size": 14})

plt.yticks(fontproperties='Times New Roman', size=14, weight="heavy")
plt.xticks(fontproperties='Times New Roman', size=14, weight="heavy")

ax.legend(handles[:2], ["Training Score", "Validation Score"])
# ax.set_title(f"Learning Curve for Random Forest")
plt.savefig("learning_curve.svg", dpi=600)
# plt.show()