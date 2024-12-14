import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


csv_path = "Fig3a_and_3b.csv"

data = pd.read_csv(csv_path)

sns.set_theme(style="ticks")
sns.color_palette("deep")
plt.figure(figsize=(7, 7))

pal = {
    "out": "black",
    "in": "red"
       }

# # Figure 3a
# fig = sns.displot(data=data,
#                   x="Eb(eV)",
#                   hue="CNT",
#                   kind="hist",
#                   palette=pal,
#                   facet_kws={"despine": False},
#                   legend=False)
#

# #Figure 3b: KDE plot
fig = sns.displot(data=data,
                  x="Eb(eV)",
                  hue="CNT",
                  kind="kde",
                  bw_adjust=1.2,
                  common_norm=False,
                  lw=2,
                  palette=pal,
                  facet_kws={"despine": False},
                  legend=False)


fig.set(xticks=[-4.0, -3.5, -3.0, -2.5])

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 14})

# plt.ylabel("Density", fontdict={"weight": "heavy", "size": 14})
# plt.xlabel("E$_{b}$ (eV)", fontdict={"weight": "heavy", "size": 14})

plt.yticks(fontproperties='Times New Roman', size=14, weight="heavy")
plt.xticks(fontproperties='Times New Roman', size=14, weight="heavy")
plt.savefig("Fig3b.png", dpi=600)
