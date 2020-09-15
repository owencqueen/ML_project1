# This file contains the functions necessary to build the b
import numpy as np
import pandas as pd
from math import pi
from matplotlib import pyplot as plt

def class1_split(x):
    # This is derived from point-slope form on our linear separation of 
    #   class 1 clusters
    return ((1.2/0.7) * x - (0.2 * 1.2) / 0.7 + 1)

def calc_cov_mat(data):
    # Calculates covariance matrix of the given data
    data_mat = np.array(data.iloc[:,0:-1])
    return np.cov(np.transpose(data_mat))

def calc_mean_vec(data):
    # Calculates mean vector of the given data
    x_col = data.iloc[:,0]
    y_col = data.iloc[:,1]

    x_mean = np.mean(x_col)
    y_mean = np.mean(y_col)

    return [x_mean, y_mean]


def multimodal_Gaussian_functions(a0 = 0.5, a1 = 0.5):
    '''
    Generates the multimodel Gaussian distributions for class 0 and class 1 in 
    the synth training dataset

    Arguments:
    ----------
    a0: float
        - a0 is multiplication factor to gauss1
        - Proportion of cluster 0 to cluster 1 in class 0
        - i.e. additive Gaussian is (a0 * gauss1) + ( (1 - a0) * gauss2)
    a1: float
        - Same as a0 but for class 1

    Returns:
    --------
    gauss_fns: list of functions
        - gauss_fns[0] is the bimodal Gaussian function for class 0
        - gauss_fns[1] is the bimodal Gaussian function for class 1

    '''
    train_data = pd.read_csv("../data/synth_train.csv")

    # First, we need to partition the data

    # Partition classes:
    class0_samples = train_data[train_data["label"] == 0]
    class1_samples = train_data[train_data["label"] == 1]

    # Initialize empty lists
    cluster0_class0 = []
    cluster1_class0 = []
    cluster0_class1 = []
    cluster1_class1 = []

    # Partition based on separation values:

    for i in range(0, class0_samples.shape[0]):
        # Partitions class 0 samples
        if (class0_samples.iloc[i, 0] < -0.3):
            cluster0_class0.append(class0_samples.iloc[i, :])
        else:
            cluster1_class0.append(class0_samples.iloc[i, :])

    cluster0_class0 = pd.DataFrame(cluster0_class0, columns = ["x", "y", "label"])
    #print(cluster0_class0)
    cluster1_class0 = pd.DataFrame(cluster1_class0, columns = ["x", "y", "label"])

    for i in range(0, class1_samples.shape[0]):
        # Partitions class 1 samples
        sample_x = class1_samples.iloc[i, 0]
        y_boundary = class1_split(sample_x)

        if (y_boundary < class1_samples.iloc[i, 1]):
            cluster0_class1.append(class1_samples.iloc[i, :])
        else:
            cluster1_class1.append(class1_samples.iloc[i, :])

    cluster0_class1 = pd.DataFrame(cluster0_class1, columns = ["x", "y", "label"])
    cluster1_class1 = pd.DataFrame(cluster1_class1, columns = ["x", "y", "label"])

    # Then, we need to calculate the means and covariances

    # class 0, means/variances
    class0_0_mu  = calc_mean_vec(cluster0_class0)
    class0_0_cov = calc_cov_mat(cluster0_class0)

    class0_1_mu  = calc_mean_vec(cluster1_class0)
    class0_1_cov = calc_cov_mat(cluster1_class0)

    # class 1, means/variances
    class1_0_mu  = calc_mean_vec(cluster0_class1)
    class1_0_cov = calc_cov_mat(cluster0_class0)

    class1_1_mu  = calc_mean_vec(cluster1_class1)
    class1_1_cov = calc_cov_mat(cluster0_class0)

    # Need to make our Gaussians to return from the function

    # Generate multimodal Gaussian for class 0:
    def class0_gaussian(x):
        # Get first Gaussian value
        det_0 = np.linalg.det(class0_0_cov)
        
        coefficient = 1 / (np.sqrt(2 * pi) * np.sqrt(det_0))

        mat_mulp = np.matmul((x - class0_0_mu), np.linalg.inv(class0_0_cov))
        mat_mulp = np.matmul(mat_mulp, np.transpose(x - class0_0_mu))
        exp_factor = np.exp( (-1 / 2.0) * mat_mulp )

        gauss1 = coefficient * exp_factor

        # Get second Gaussian value:
        det_1 = np.linalg.det(class0_1_cov)
        
        coefficient = 1 / (np.sqrt(2 * pi) * np.sqrt(det_1))

        mat_mulp = np.matmul((x - class0_1_mu), np.linalg.inv(class0_1_cov))
        mat_mulp = np.matmul(mat_mulp, np.transpose(x - class0_1_mu))
        exp_factor = np.exp( (-1 / 2.0) * mat_mulp )

        gauss2 = coefficient * exp_factor

        return ((a0 * gauss1) + ( (1 - a0) * gauss2))

    # Generate multimodal Gaussian for class 1:
    def class1_gaussian(x):

        det_0 = np.linalg.det(class1_0_cov)
        
        coefficient = 1 / (np.sqrt(2 * pi) * np.sqrt(det_0))

        mat_mulp = np.matmul((x - class1_0_mu), np.linalg.inv(class1_0_cov))
        mat_mulp = np.matmul(mat_mulp, np.transpose(x - class1_0_mu))
        exp_factor = np.exp( (-1 / 2.0) * mat_mulp )

        gauss1 = coefficient * exp_factor

        # Get second Gaussian value:
        det_1 = np.linalg.det(class1_1_cov)
        
        coefficient = 1 / (np.sqrt(2 * pi) * np.sqrt(det_1))

        mat_mulp = np.matmul((x - class1_1_mu), np.linalg.inv(class1_1_cov))
        mat_mulp = np.matmul(mat_mulp, np.transpose(x - class1_1_mu))
        exp_factor = np.exp( (-1 / 2.0) * mat_mulp )

        gauss2 = coefficient * exp_factor

        return ((a1 * gauss1) + ( (1 - a1) * gauss2))

    return [class0_gaussian, class1_gaussian]

def plot_countours(class_number = 0):
    '''
    Plots the contours for bimodal Gaussian distribution for given class in
    synth dataset

    Arguments:
    ----------
    class_number: int
        - Specifies which class to run the contour plot for
        - Options: 0 or 1

    No explicit return
    '''
    # Plots the contours of the multimodal Gaussian distributions for class 0 or 1
    normal0, normal1 = multimodal_Gaussian_functions()

    # Generate values to evaluate:
    x = np.linspace(-2, 1.4, 50)
    y = np.linspace(-0.3, 1.4, 50)

    X, Y = np.meshgrid(x, y)

    xflat = X.ravel()
    yflat = Y.ravel()

    both = np.c_[xflat, yflat]

    fxy = []
    for xy in both: # Evaluates each point
        # Change the below to "normal1" to see class 1 bimodal distribution
        if (class_number == 0):
            fxy.append(normal0(xy))
        else:
            fxy.append(normal1(xy))

    # Reshape to work with mesh
    fxy = np.array(fxy)
    fxy = fxy.reshape(X.shape)

    # Plot it:
    plt.xlabel("x")
    plt.ylim(-0.3, 1.4)
    plt.xlim(-1.5, 1.2)
    plt.ylabel("y")
    plt.title("Class 1 Bimodal Distribution Contours - Synth Dataset; a = 0.5")
    plt.contourf(X, Y, fxy, 25)
    plt.show()

if __name__ == "__main__":
    plot_countours(class_number = 0)

