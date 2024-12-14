import joypy
import matplotlib.pyplot as plt
import pandas as pd


csv_path = "Figure5a.csv"
png_path = "d-band-center-O.png"

data = pd.read_csv(csv_path)

joypy.joyplot(data,
              column=["d band center in CNT", "d band center out CNT"],
              by="Number of adsorbates",
              fade=True,
              legend=False,
              ylim="own",
              color=["red", "black"]
              )


plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 11})
plt.yticks(fontproperties='Times New Roman', size=11)
plt.xticks(fontproperties='Times New Roman', size=11)
plt.savefig(png_path, dpi=600)
