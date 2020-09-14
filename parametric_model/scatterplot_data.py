import pandas as pd
from matplotlib import pyplot as plt


def scatterplot_for_synth(type_data, show = True, use_pd_df = False):
    '''
    Generates scatterplot for synthetic data

    Arguments:
    ----------
    type_data: string or pandas dataframe
        - Options: "train", "test"
        - Specifies which data to plot
        - If it's a pandas dataframe already, use_pd_df must be True
    show: bool, optional
        - If true, shows the ploit
        - If false, does not show the plot but the plt object is saved
            in the local environment
    use_pd_df: bool, optional
        - If true, type_data is a pandas dataframe

    Returns:
    --------
    No explicit return, but the scatterplot is output if show == True
    '''

    if (use_pd_df):
        synth_df = type_data
    else:
        if (type_data == "train"):
            synth_df = pd.read_csv("../data/synth_train.csv")
        elif (type_data == "test"):
            synth_df = pd.read_csv("../data/synth_test.csv")

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