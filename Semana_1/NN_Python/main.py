import Data_Process
import NN_utils

"""
    Main file test
"""
if __name__ == '__main__':
    labels, data = Data_Process.readDatabase('C:/Users/Diogo/Desktop/Semana_1/Database_Balanced.csv')

    dataset = []
    dataset.append(data)
    dataset.append(labels)

    training_set_values, training_set_labels, test_set_values, test_set_labels = Data_Process.dataSplit(
        dataset, 0.6, 0.4
    )

    scaled_train_set = Data_Process.dataNormalizing(training_set_values)
    scaled_test_set = Data_Process.dataNormalizing(test_set_values)

    randomSearch_tuner, rand_time = NN_utils.randomSearch_tuning(scaled_train_set, training_set_labels,
                                                                 scaled_test_set, test_set_labels)

    hyperband_tuner, hyper_time = NN_utils.hyperband_tuning(scaled_train_set, training_set_labels,
                                                            scaled_test_set, test_set_labels)
    # Print results
    print(randomSearch_tuner.results_summary())
    print("Random Search execution time: ", rand_time, "seconds")
    print(hyperband_tuner.results_summary())
    print("Hyperband execution time: ", hyper_time, "seconds")
