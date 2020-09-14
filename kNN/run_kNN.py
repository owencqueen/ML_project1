import pandas as pd
from kNN import kNN_classifier

# Set dataset variable equal to "synth" in order to run synth dataset
dataset = "synth"

train = pd.read_csv("../data/" + dataset + "_train.csv")

model = kNN_classifier(train)

model.classify(dataset + "_test.csv", k = 13)

# Run the classification procedure on each
#for i in range(1, 15):
#    model.classify(dataset + "_test.csv", k = i)

model.plot_overall_acc()
