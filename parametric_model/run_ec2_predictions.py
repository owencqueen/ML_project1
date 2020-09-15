import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MD_classifier import classifier

# Run the model on synth dataset:
# ------------------------------
synth_tr = pd.read_csv("../data/synth_train.csv")

model = classifier(synth_tr)

# We will run the bimodal distribution:
model.classify( test_data = 'synth_test.csv', discriminant_type = "bimodal")

