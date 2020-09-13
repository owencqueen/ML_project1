import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

class kNN_classifier:

    def __init__(self, train_data):

        self.train = train_data

        self.num_features = train_data.shape[1] - 1

        labels = train_data.iloc[:,-1]
        labels = set(labels)
        self.labels = list(labels) # Get individual labels
        self.labels = [int(i) for i in self.labels]

        self.num_samples = train_data.shape[0]

        self.overall_acc_stats = {} # Initialize dictionary to keep up with accuracies

    def classify(self, testing_data, k, show_statistics = True, df_provided = False, progress_bar = False):

        start_time = time.time() # Start the clock

        if (df_provided):
            test = testing_data
        else:
            test = pd.read_csv('../data/' + testing_data)

        self.predictions = []

        dtype = [("dist", float), ("label", int)]

        run = 0
        total_runs = test.shape[0]

        for i in range(total_runs):

            x = test.iloc[i, 0:-1]

            dists = []

            # Calculate all euclidean distances:
            for sample in range(self.num_samples):
                sample_x = self.train.iloc[sample,0:-1]
                sample_l = self.train.iloc[sample, -1]
                dists.append( (euclidean_dist(sample_x, x), sample_l) )    
                    
            conj_dists = np.array(dists, dtype = dtype)  # Create structured array with labels
            conj_dists = np.sort(conj_dists, order = 'dist') # Sort structured array based on distance

            k_shortest = conj_dists[0:k] # Get k smallest dists

            k_labels = []

            for i in range(len(k_shortest)):
                k_labels.append(k_shortest[i][1])

            label_counts = [0] * len(self.labels) # Will be used to count frequencies

            # Take majority vote:
            for label in k_labels:
                label_counts[self.labels.index(label)] += 1

            max_label_ind = label_counts.index(max(label_counts))
            #print(self.labels[max_label_ind])
            self.predictions.append(self.labels[max_label_ind])

            if (progress_bar): # Print the progress bar if specified
                print("{} of {} runs".format(run, total_runs))
                #if ((run / total_runs * 100) % 10 == 0):
                #    print ("{} percent finished".format(int(run / total_runs * 100)))
            run += 1
            #run += 1

        if (show_statistics):
            self.accuracy_stats(test, k = k, save_overall = True) # Prints statistics for classification algorithm
            print("k = {} Runtime: {} seconds".format(k, time.time() - start_time))
            print("") # Need newline

    def accuracy_stats(self, test_data, k, save_overall = True):
        # Need:
        #   1. overall classification accuracy
        #   2. classwise accuracy - all classes
        #   3. run time - printed in classify function

        # Overall classification accuracy:
        true_labels = list(test_data.iloc[:, -1])

        overall_correct = 0
        class_wise_correct = [0] * len(self.labels)
        class_wise_total = [0] * len(self.labels)
        # For classwise accuracy, numbers will be stored in 

        for i in range(0, len(true_labels)):

            predict = self.predictions[i]

            if (true_labels[i] == predict):
                overall_correct += 1 # Add to correct count if they match

                # Add to class-wise accuracy
                class_wise_correct[self.labels.index(predict)] += 1

            class_wise_total[self.labels.index(predict)] += 1

        overall_acc = overall_correct / len(true_labels)
        class_wise_accuracy = [(class_wise_correct[i] / class_wise_total[i]) \
                                for i in range(0, len(self.labels))]

        print("Overall Accuracy:    {acc:.5f}".format(acc = overall_acc))

        for i in range(0, len(self.labels)):
            print("Classwise accuracy for \"{cl}\" class: {acc:.5f}"\
                    .format(cl = self.labels[i], acc = class_wise_accuracy[i]))

        if (save_overall):
            self.overall_acc_stats[k] = overall_acc 

    def plot_overall_acc(self):
        x = self.overall_acc_stats.keys()
        y = self.overall_acc_stats.values()

        plt.bar(x, y)
        plt.xlabel("k value")
        plt.ylabel("Overall Classification Accuracy")
        plt.ylim(0.5, 1)
        plt.title("Classification Accuracy vs. k")
        plt.show()

    def plot_boundaries(self):
        # Plots the decision boundary for kNN
        # Note: only for use in synth dataset
        # Hardcoded to work with only synth dataset

        #point_colors = colors.ListedColormap(["red", "blue"])
        area_colors  = colors.ListedColormap(["salmon", "lightskyblue"])

        labeled_lists = [ [], [] ] 

        # Plot the points with corresponding colors:
        for i in range(0, self.train.shape[0]):
            label = int(self.train.iloc[i, -1])
            labeled_lists[label].append(list(self.train.iloc[i, :]))

        # Get the labels for each of the training data points
        label0_x = [i[0] for i in labeled_lists[0]]
        label0_y = [i[1] for i in labeled_lists[0]]

        label1_x = [i[0] for i in labeled_lists[1]]
        label1_y = [i[1] for i in labeled_lists[1]]

        # Need to assign a color to each point in the mesh:
        min_x = self.train.iloc[:, 0].min() - 1
        max_x = self.train.iloc[:, 0].max() + 1

        min_y = self.train.iloc[:, 1].min() - 1
        max_y = self.train.iloc[:, 1].max() + 1

        # There will be a 0.03 resolution in our mesh
        mesh_resolution = 0.03

        xlist = np.arange(min_x, max_x, mesh_resolution)
        ylist = np.arange(min_y, max_y, mesh_resolution)

        # Create mesh
        xmesh, ymesh = np.meshgrid(xlist, ylist)

        # Need to get new predictions based on each point in the mesh:
        dict_data = {'x': xmesh.ravel(), 'y': ymesh.ravel(), "label": ([0] * len(xmesh.ravel()))}
        mesh_data = pd.DataFrame(dict_data)

        # Classify each of the points:
        self.classify(testing_data = mesh_data, k = 13, show_statistics = False, \
                        df_provided = True, progress_bar = True)
        preds = np.array(self.predictions)

        # Shape our predictions to fit the mesh data points
        preds = preds.reshape(xmesh.shape)

        # Plot each of the points in the mesh (effectively decision boundary)
        plt.figure()
        plt.pcolormesh(xmesh, ymesh, preds, cmap = area_colors)

        # Plot the original training data points
        plt.scatter(label0_x, label0_y, c = "red")  
        plt.scatter(label1_x, label1_y, c = "blue")

        plt.xlabel("x")
        plt.ylabel('y')
        plt.title("kNN (k = 13) Decision Boundary on Synth Dataset")
        
        plt.show()
        

def euclidean_dist(x_p, x_i):
    # Get the list of norms for vector each other one
    x_p = np.array(x_p)
    x_i = np.array(x_i)
    return np.linalg.norm(x_p - x_i)