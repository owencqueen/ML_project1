#!/usr/bin/env python3
import pandas as pd
from kNN import kNN_classifier

dataset = "synth"
train = pd.read_csv("../data/" + dataset + "_train.csv")

model = kNN_classifier(train)

model.plot_boundaries()