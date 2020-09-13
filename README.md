# Project 1
Project 1 for COSC 522 - Machine Learning - at UTK

This README will describe how to run the code associated with this project. Please see the report for an in-depth analysis of the results of these models.
The project writeup can be found at [this link](http://web.eecs.utk.edu/~hqi/cosc522/project/proj1.htm). 

### Section 1
The files for this portion are contained in the parametric_model directory.

#### Part A
The code for this portion is in the parametric_model/scatterplot_data.py file. All you need to do to generate the scatterplot (shown below) is to run this script.

#### Part B
For this part, run the parametric_model/run_classifier.py script. This will also output the decision boundaries (for Section 1, part D) after printing the accuracy statistics for each discriminant function on the synth dataset. You must then close out of the decision boundary plot, and then the script will run the classifier on the Pima dataset and print the accuracy statistics.

#### Part C
No code is associated with this part.

#### Part D
This plot (shown below) will be output to the screen when running the parametric_model/run_classifier.py script.

#### Part E
On the report, I generated 4 separate plots for this part. To generate the plots shown to the right on each section, you will just run the script parametric_model/run_prior_prob.py. In order to generate the plots shown to the left, this will require changing the code in parametric_model/MD_classifier.py. You will have to go to this file and uncomment line 247 while commenting line 248. Then, the plots to the left should show up after running the parametric_model/run_prior_prob.py script.

### Section 2
All of the code for this section is contained in the kNN directory.

#### Part A
To generate similar output to that provided by parametric_model/run_classifier.py, run the script kNN/run_kNN.py. This outputs given accuracy statistics and runtimes for a k values ranging from 1 - 14. In order to specify whether to run the classifier for "synth" or "Pima" data, set the "dataset" variable (as explained in the comments).

#### Part B

#### Part C

#### Part D
