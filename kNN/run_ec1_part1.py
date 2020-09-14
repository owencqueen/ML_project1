import numpy as np
import pandas as pd
from kNN import kNN_classifier

# Set up the model
dataset = "pima"

train = pd.read_csv("../data/" + dataset + "_train.csv")

model = kNN_classifier(train)

# First, we want to calculate the proportions of sample 
#   labels in the training dataset
freqs = train.iloc[:,-1].value_counts()
freq_0 = freqs[0]
freq_1 = freqs[1]

total = freq_0 + freq_1 # Get total number of samples in training dataset

# To run the directly-proportional prior probabilities, uncomment the line below:
prior_probs = {0: freq_0 / total, 1: freq_1 / total}

# To run the inversely-proportional prior probabilities, uncomment the line below:
#prior_probs = {0: freq_1 / total, 1: freq_0 / total}

# Varying the k value with the same probability:
for i in [2, 4, 6, 8, 10, 12]:
    model.classify(dataset + "_test.csv", k = i, pp_weights = prior_probs)

# Plots the bar graph of overal accuracy vs. k
model.plot_overall_acc(plot_pp = True, pp = prior_probs)
