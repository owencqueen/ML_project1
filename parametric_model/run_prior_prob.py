import numpy as np
import pandas as pd
from MD_classifier import classifier

# Vary the prior probabilities
#   Will test 25 different values from 0 - 1
prior_probs = np.linspace(0, 1, 25)

# Change the value of dataset_name in order to run for corresponding training datasets
dataset_name = "pima_train.csv"

train_data = pd.read_csv("data/" + dataset_name)
model = classifier(train_data)

for pp in prior_probs:
    model.classify( test_data = dataset_name, discriminant_type = "euclidean", prior_probs = [pp, (1 - pp)], plot_predictions = False)
    model.classify( test_data = dataset_name, discriminant_type = "mahalanobis", prior_probs = [pp, (1 - pp)], plot_predictions = False)
    model.classify( test_data = dataset_name, discriminant_type = "quadratic", prior_probs = [pp, (1 - pp)], plot_predictions = False)

model.plot_class_0_acc(prior_probs) 