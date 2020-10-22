import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import os.path

"""
    Read csv file from a given path
"""


def readDatabase(path):
    database = pd.read_csv(path, delimiter=';')
    # Get features values
    data = database.drop('class', axis=1)  # Drops label column
    data = data.values  # Returns a numpy array representing dataframe

    # Get labels
    labels = database['class'].values

    return labels, data


"""
    Split dataset into two subsets, where the subsets sizes is set by percentage (range [0, 1])
"""


def dataSplit(dataset, train_perc, test_perc):
    assert 0 <= train_perc <= 1 and 0 <= test_perc <= 1 and train_perc + test_perc == 1
    training_set_values, test_set_values, training_set_labels, test_set_labels = train_test_split(
        dataset[0], dataset[1], test_size=test_perc, shuffle=True, stratify=dataset[1])
    return training_set_values, training_set_labels, test_set_values, test_set_labels


"""
    Normalize data given as input, scales the values of each feature to the range [-1, 1]
"""


def dataNormalizing(data):
    if os.path.isfile("venv/scaler.save"):
        scaler = joblib.load("venv/scaler.save")
    else:
        scaler = preprocessing.MaxAbsScaler().fit(data)
        joblib.dump(scaler, "venv/scaler.save")

    dataScaled = scaler.transform(data)
    return dataScaled


labels, data = readDatabase('C:/Users/Diogo/Desktop/Semana_1/Database_Balanced.csv')

dataset = []
dataset.append(data)
dataset.append(labels)

training_set_values, training_set_labels, test_set_values, test_set_labels = dataSplit(
    dataset, 0.6, 0.4
)

print("Data split")
scaledData = dataNormalizing(training_set_values)
print("Normalization")

