import pandas as pd
from matplotlib import pyplot as plt

# Generates scatterplot for synthetic data
def scatterplot_for_synth(type_data, show, use_pd_df = False):
    
    if (use_pd_df):
        synth_df = type_data
    else:
        if (type_data == "train"):
            synth_df = pd.read_csv("data/synth_train.csv")
        elif (type_data == "test"):
            synth_df = pd.read_csv("data/synth_test.csv")

    groups = synth_df.groupby("label")
    for name, group in groups:
        if (name == 1):
            name = 1
        else:
            name = 0
        plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name)
    
    plt.title("Synthetic Data Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    
    if (show):
        plt.legend()
        plt.show()

if __name__ == "__main__":
    scatterplot_for_synth("test", show = True)