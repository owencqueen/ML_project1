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

model = kNN_classifier(train)

# Vary the prior probability:
freq_0_varying = np.linspace(prior_probs[0] - 0.3, prior_probs[0] + 0.3, 8)

for f_0 in freq_0_varying:
    # Run the classification procedure on each
    prior_probs = {0: f_0, 1: (1 - f_0)}
    print("prior prob 0: ", prior_probs[0])
    for i in [10, 12, 13, 14, 15]:
        model.classify(dataset + "_test.csv", k = i, pp_weights = prior_probs)

# Plotting function
model.plot_overall_acc_w_varying(pp_varied = freq_0_varying)