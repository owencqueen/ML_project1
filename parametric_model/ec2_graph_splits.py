# This file plots the splits on the data for the two classes

from matplotlib import pyplot as plt
from scatterplot_data import scatterplot_for_synth

# Plot the training data
scatterplot_for_synth(type_data = "train", show = False)

# Plot the splits between clusters
plt.axvline(-0.3, c = 'green', linestyle = '-', label = "Class 0 Boundary")
plt.plot([-0.5, 0.2], [-0.2, 1], c = 'red', linestyle = '-', label = "Class 1 Boundary")

plt.legend()
plt.show()
