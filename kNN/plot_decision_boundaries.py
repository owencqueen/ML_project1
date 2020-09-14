#!/usr/bin/env python3
import pandas as pd
from kNN import kNN_classifier

dataset = "synth"
train = pd.read_csv("../data/" + dataset + "_train.csv")

model = kNN_classifier(train)

# Plots the boundaries for k = 13 (found to be optimal k value on synth dataset)plp
model.plot_boundaries(k = 13, mesh_resolution = 0.03)
