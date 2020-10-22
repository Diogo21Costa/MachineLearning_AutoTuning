import pandas as pd


def readDatabase(path):
    database = pd.read_csv(path)

    # Get features values
    # data = database.drop('label', axis=1)  # Drops label column
    data = data.values  # Returns a numpy array representing dataframe

    # Get labels
    labels = database['label'].values

    return labels, data


labels, data = readDatabase("C:/Users/Diogo/Desktop/Semana_1/Database_Balanced.csv")
