# Reads all of the data and performs preprocessing
import pandas as pd
import numpy as np

def read_data_for_synth(file_name, export_fname):

    synth_df = pd.DataFrame(columns = ['x', 'y', 'label'])
    with open(file_name) as f:
        for l in f:
            line = l.split()
            data_line = {'x': float(line[0]), 'y': float(line[1]), 'label': int(line[2])}
            
            synth_df = synth_df.append(data_line, ignore_index = True, sort = False)

    synth_df.to_csv("data/" + export_fname)
    print(synth_df)

def read_data_for_pima(file_name, export_fname):

    pima_df = pd.DataFrame(columns = ['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age', 'type'])
    with open(file_name) as f:
        for l in f:
            line = l.split()
            
            # Change from "Yes" and "No" to 0 and 1, respectively
            #last_col = 0 if (data_line[7] == 'Yes') else 1 

            data_line = {'npreg': int(line[0]), 
                         'glu': int(line[1]), 
                         'bp': int(line[2]),
                         'skin': int(line[3]),
                         'bmi': float(line[4]),
                         'ped': float(line[5]),
                         'age': int(line[6]),
                         'type': 0 if (line[7] == 'Yes') else 1}
            
            pima_df = pima_df.append(data_line, ignore_index = True, sort = False)

    # Need to normalize all of the columns (features)
    for i in range(0, len(pima_df.columns) - 1):

        column_i = pima_df.iloc[:, i] # Get ith col

        # Compute standard deviation of the ith column
        std_i = np.std(column_i)
        
        # Compute mean of the ith column
        mu_i = np.mean(column_i)

        # Normalize and set the columns
        pima_df.iloc[:, i] = [((x - mu_i) / std_i) for x in pima_df.iloc[:, i]]

    pima_df.to_csv("data/" + export_fname)
    print(pima_df)

read_data_for_synth(file_name = "data/synth.tr.txt", export_fname = "synth_train.csv")
read_data_for_synth(file_name = "data/synth.te.txt", export_fname = "synth_test.csv")
read_data_for_pima(file_name = "data/pima.tr.txt", export_fname = "pima_train.csv")
read_data_for_pima(file_name = "data/pima.te.txt", export_fname = "pima_test.csv")
