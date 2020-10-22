# Neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Hyperparameteres tuning
import kerastuner as kt
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

# Time measurements
import time

"""
    Model setup with hyperparameters auto tuning
"""


def createModel(hp):
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.Dense(hp.Int("input_units", min_value=8, max_value=256, step=8)))
    model.add(tf.keras.layers.Activation(hp.Choice('dense_activation',
                                                   ['relu', 'tanh', 'sigmoid'],
                                                   default='relu')))

    # Hidden layers
    for i in range(hp.Int("n_layers", min_value=1, max_value=8)):
        model.add(tf.keras.layers.Dense(hp.Int(f"dense_{i}_units", min_value=32, max_value=256, step=32)))
        model.add(tf.keras.layers.Activation(hp.Choice('dense_activation',
                                                       ['relu', 'tanh', 'sigmoid'],
                                                       default='relu')))

    # Output layer
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Activation('softmax'))

    # Compile model
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(dataset[0], dataset[1], epochs=5)

    return model


def randomSearch_tuning(scaled_train_set, training_set_labels,
                        scaled_test_set, test_set_labels):

    randomSearch_tuner = RandomSearch(createModel,
                                      objective="val_accuracy",
                                      max_trials=10,
                                      executions_per_trial=2,
                                      directory="random_path")

    random_start_time = time.time()
    randomSearch_tuner.search(x=scaled_train_set,
                              y=training_set_labels,
                              epochs=10,
                              batch_size=64,
                              validation_data=(scaled_test_set, test_set_labels))

    rand_time = time.time() - random_start_time

    return randomSearch_tuner, rand_time


def hyperband_tuning(scaled_train_set, training_set_labels,
                     scaled_test_set, test_set_labels):
    hyperband_tuner = kt.Hyperband(createModel,
                                   objective='val_accuracy',
                                   max_epochs=10,
                                   factor=3,
                                   directory="hyper_path")

    hyper_start_time = time.time()
    hyperband_tuner.search(x=scaled_train_set,
                           y=training_set_labels,
                           epochs=10,
                           validation_data=(scaled_test_set, test_set_labels))
    hyper_time = time.time() - hyper_start_time

    return hyperband_tuner, hyper_time
