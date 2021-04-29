import os, json, csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


plot_format = {
    "markersize": 4,
    "color": "black",
    "linestyle": "-",
    "linewidth": 1.1
}
label_format = {
    "fontname": "Times New Roman",
    "fontsize": 14
}
legend_format = {
    "prop": font_manager.FontProperties(family="Times New Roman")
}

os.chdir("C:\\Users\\ben\Desktop\\gitrepos\\physics-project\\analysis\\data")

rmse = {}

with open("rmse.json", 'r') as file:
    rmse = json.load(file)

for i in range(1, 4):
    plt.figure(figsize=(10, 10))
    plt.xlabel("Applied Magnetic Field / T", **label_format)
    plt.ylabel("RMSE", **label_format)
    plt.plot(rmse[str(i)]['field1'], rmse[str(i)]['model1'], label=rmse[str(i)]['model1_name'][0], **plot_format)
    plt.plot(rmse[str(i)]['field2'], rmse[str(i)]['model2'], label=rmse[str(i)]['model2_name'][0], **plot_format)
    plt.legend(**legend_format)
    plt.show()
