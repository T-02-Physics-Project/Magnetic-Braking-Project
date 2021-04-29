import os, json, csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


plot_format = {
    "markersize": 4,
    "color": "black",
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

with open("rmse.json", 'r') as file:
    rmse = json.load(file)

with open('mpe.json', 'r') as file:
    mpe = json.load(file)

for i in range(1, 4):
    data = rmse[str(i)]
    plt.figure(figsize=(10, 10))
    plt.xlabel("Applied Magnetic Field / T", **label_format)
    plt.ylabel("RMSE", **label_format)
    plt.plot(data['field1'], data['model1'], "k--", label=data['model1_name'], **plot_format)
    plt.plot(data['field2'], data['model2'], "k-", label=data['model2_name'], **plot_format)
    plt.plot(data['field3'], data['model3'], "k", linestyle='dashdot', label=data['model3_name'], **plot_format)
    plt.legend(**legend_format)
    plt.show()

for i in range(1, 4):
    data = mpe[str(i)]
    plt.figure(figsize=(10, 10))
    plt.xlabel("Applied Magnetic Field / T", **label_format)
    plt.ylabel("MPE", **label_format)
    plt.plot(data['field1'], data['model1'], "k--", label=data['model1_name'], **plot_format)
    plt.plot(data['field2'], data['model2'], "k-", label=data['model2_name'], **plot_format)
    plt.plot(data['field3'], data['model3'], "k", linestyle='dashdot', label=data['model3_name'], **plot_format)
    plt.legend(**legend_format)
    plt.show()
