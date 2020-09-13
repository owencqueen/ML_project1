import pandas as pd
from kNN import kNN_classifier

dataset = "pima"
train = pd.read_csv("data/" + dataset + "_train.csv")

model = kNN_classifier(train)
#model.classify('synth_test.csv', k = 3)

for i in range(1, 15):
    model.classify(dataset + "_test.csv", k = i)

model.plot_overall_acc()
