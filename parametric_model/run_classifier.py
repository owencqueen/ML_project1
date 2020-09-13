import pandas as pd
from MD_classifier import classifier

# Run the model on synth dataset:
# ------------------------------
synth_tr = pd.read_csv("../data/synth_train.csv")

model = classifier(synth_tr)

# We will run part B of Question 1 (equal prior probabilities):
model.classify( test_data = 'synth_test.csv', discriminant_type = "euclidean", plot_predictions = True)
model.classify( test_data = 'synth_test.csv', discriminant_type = "mahalanobis", plot_predictions = True)
model.classify( test_data = 'synth_test.csv', discriminant_type = "quadratic", plot_predictions = True)

# Plot the decision boundaries on synth dataset
model.plot_decision_boundaries(show = True, dis_fn = "all")

# Run the model on pima dataset:
# ------------------------------
pima_tr = pd.read_csv("../data/pima_train.csv")

model = classifier(pima_tr)

# We will run part B of Question 1 (equal prior probabilities):
model.classify( test_data = 'pima_test.csv', discriminant_type = "euclidean")
model.classify( test_data = 'pima_test.csv', discriminant_type = "mahalanobis")
model.classify( test_data = 'pima_test.csv', discriminant_type = "quadratic")

