'''
Contains methods for dataset parsing, normalizing and reading
'''

import numpy as np

def _raw_data(filename):
    # read the data
    vals = np.genfromtxt(filename, delimiter = ",")
    
    # shuffle values
    np.random.shuffle(vals)
    
    # skip rows with missing values
    select = np.sum(np.isnan(vals), axis = 1) == 0
    vals = vals[select, :]
        
    # normalize dataset
    mean = np.mean( vals , axis = 0)
    dev = np.std( vals, axis = 0)
    
    vals[:,:-1] = (vals[:,:-1] - mean[:-1]) / dev[:-1];
    
    resX = vals[:, :-1]
    resY = vals[:, -1]
    # thing below is just a dummy, ignore it
    resM = resY*0;
    
    InputSize = resX.shape[1]
    OutputSize = 1
    
    return {
            'X': resX, 'Y': resY, 'M':resM,
            "InputSize":InputSize, "OutputSize":OutputSize,
            "Mean":mean, "Deviation":dev
            }

def SplitData(dataset, datasetSplit):
        
        X = dataset["X"]
        Y = dataset["Y"]
        M = dataset["M"]
        mean = dataset["Mean"]
        dev = dataset["Deviation"]
        InputSize = dataset["InputSize"]
        OutputSize = dataset["OutputSize"]
        
        endTrain = int(X.shape[0]*datasetSplit[0])
        endVal = endTrain + int(X.shape[0]*datasetSplit[1])
        
        dataTrain = {"X":X[:endTrain,], "Y":Y[:endTrain,], "M": M[:endTrain,], 
                  "InputSize":InputSize, "OutputSize":OutputSize,
                  "Mean":mean, "Deviation":dev}
        
        dataVal = {"X":X[ endTrain:endVal,], "Y":Y[ endTrain:endVal,], "M": M[ endTrain:endVal,], 
                  "InputSize":InputSize, "OutputSize":OutputSize,
                  "Mean":mean, "Deviation":dev}
        
        dataTest = {"X":X[endVal:,], "Y":Y[ endVal:,], "M": M[endVal:,], 
                  "InputSize":InputSize, "OutputSize":OutputSize,
                  "Mean":mean, "Deviation":dev}
        
        return dataTrain, dataVal, dataTest 
    
def abalone_dataset(split):
    return SplitData(_raw_data("abalone.csv"), split);


def chess_dataset(split):
    return SplitData(_raw_data("chess.csv"), split);