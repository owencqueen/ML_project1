import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MD_classifier import classifier

# Load model:
synth_tr = pd.read_csv("../data/synth_train.csv")

model = classifier(synth_tr)

# I have modified these values from the report to decrease runtime:
varying_0_vals = np.linspace(0.3, 0.8, 8)
varying_1_vals = np.linspace(0.3, 0.8, 8)

overall_acc = np.zeros(shape = (len(varying_0_vals), len(varying_1_vals)))

max_acc = 0
max_i = 0
max_j = 0

# Iterate over possible a values for each class
for i in range(0, len(varying_0_vals)):

    for j in range(0, len(varying_1_vals)):
        overall_acc[i][j] = model.classify( test_data = 'synth_test.csv', discriminant_type = "bimodal", \
            a_vals = [varying_0_vals[i], varying_1_vals[j]])
        
        if(overall_acc[i][j] > max_acc):
            max_acc = overall_acc[i][j]
            max_i = i
            max_j = j

val1, val2 = np.meshgrid(varying_0_vals, varying_1_vals)

# Plot the accuracies of each of these runs of the bimodal distribution
sizes = [100 * (i**20) for i in overall_acc.ravel()]
sizes = np.array(sizes)

# Plotting the points with the given sizes
plt.scatter(val1.ravel(), val2.ravel(), s = sizes) # Size of each point corresponds to accuracy w/ those variables
plt.xlabel("a_0 value")
plt.ylabel("a_1 value")
plt.title("Accuracy for Given a Values (size denotes accuracy)")
plt.show()