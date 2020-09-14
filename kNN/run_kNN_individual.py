import pandas as pd
from kNN import kNN_classifier

# Set dataset variable equal to "synth" in order to run synth dataset
dataset = "synth"

train = pd.read_csv("../data/" + dataset + "_train.csv")

model = kNN_classifier(train)

# For Synth dataset, run the command below:
model.classify(dataset + "_test.csv", k = 13)

#For Pima dataset, run the command below:
#model.classify(dataset + "_test.csv", k = 14)