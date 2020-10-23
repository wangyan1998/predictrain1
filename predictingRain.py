from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt


def load_data(path, test_split=0.2, seed=113):
    x = np.loadtxt(path, usecols=(0, 1, 2, 3, 4, 5))
    y = np.loadtxt(path, usecols=6)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]
    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


(train_dataBD, train_targetsBD), (test_dataBD, test_targetsBD) = load_data(path='train.txt')
(train_dataGPS, train_targetsGPS), (test_dataGPS, test_targetsGPS) = load_data(path='trainGPS.txt')

mean = train_dataBD.mean(axis=0)
train_dataBD -= mean
std = train_dataBD.std(axis=0)
train_dataBD /= std

test_dataBD -= mean
test_dataBD /= std

mean = train_dataGPS.mean(axis=0)
train_dataGPS -= mean
std = train_dataGPS.std(axis=0)
train_dataGPS /= std

test_dataGPS -= mean
test_dataGPS /= std


def build_model(train_data):
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_dataBD) // k

# num_epochs = 100
# all_scores = []
# for i in range(k):
#     print('processing fold #', i)
#     # Prepare the validation data: data from partition # k
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#
#     # Prepare the training data: data from all other partitions
#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples],
#          train_data[(i + 1) * num_val_samples:]],
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i + 1) * num_val_samples:]],
#         axis=0)
#
#     # Build the Keras model (already compiled)
#     model = build_model()
#     # Train the model (in silent mode, verbose=0)
#     model.fit(partial_train_data, partial_train_targets,
#               epochs=num_epochs, batch_size=1, verbose=0)
#     # Evaluate the model on the validation data
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
#
# np.mean(all_scores)

# Some memory clean-up
K.clear_session()

num_epochs = 3000
all_mae_histories_bd = []
for i in range(k):
    print('processing BD fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_dataBD[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targetsBD[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_dataBD[:i * num_val_samples],
         train_dataBD[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targetsBD[:i * num_val_samples],
         train_targetsBD[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model(train_dataBD)
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories_bd.append(mae_history)

all_mae_histories_gps = []
for i in range(k):
    print('processing GPS fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_dataGPS[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targetsGPS[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_dataGPS[:i * num_val_samples],
         train_dataGPS[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targetsGPS[:i * num_val_samples],
         train_targetsGPS[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model(train_dataGPS)
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories_gps.append(mae_history)

average_mae_history_bd = [
    np.mean([x[i] for x in all_mae_histories_bd]) for i in range(num_epochs)]

average_mae_history_gps = [
    np.mean([x[i] for x in all_mae_histories_gps]) for i in range(num_epochs)]


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history_bd = smooth_curve(average_mae_history_bd[:])
smooth_mae_history_gps = smooth_curve(average_mae_history_gps[:])

plt.plot(range(1, len(smooth_mae_history_bd) + 1), smooth_mae_history_bd, 'g',
         range(1, len(smooth_mae_history_gps) + 1), smooth_mae_history_gps, 'r')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# # Get a fresh, compiled model.
# model = build_model(train_dataBD)
# # Train it on the entirety of the data.
# model.fit(train_dataBD, train_targetsBD,
#           epochs=80, batch_size=16, verbose=0)
# test_mse_scoreBD, test_mae_scoreBD = model.evaluate(test_dataBD, test_targetsBD)
#
# # Get a fresh, compiled model.
# model = build_model(train_dataGPS)
# # Train it on the entirety of the data.
# model.fit(train_dataGPS, train_targetsGPS,
#           epochs=80, batch_size=16, verbose=0)
# test_mse_scoreGPS, test_mae_scoreGPS = model.evaluate(test_dataGPS, test_targetsGPS)
