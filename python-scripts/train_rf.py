import plac
import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os

# Average window_stride elements together to form a single row
WINDOW_STRIDE = 12

SAMPLE_HOURS = WINDOW_STRIDE / 12.0

# Number of future samples to mean for prediction
PREDICTION_WINDOW = int(24 / SAMPLE_HOURS)

# Length of the windowed sequence
SEQUENCE_LENGTH = int(7 * 24 / SAMPLE_HOURS)

# Input Features
INPUT_COLUMNS = ['epoch', 'day_of_year', 'hour', 'temp', 'windspd', 'winddir', 'wind_x_dir', 'wind_y_dir', 'no', 'no2',
                 'nox', 'o3']
OUTPUT_COLUMNS = ['no', 'no2', 'nox', 'o3']

# Fit the sequence to y = mx+b and add the coeff / intercept
REGRESSION_FEATURES = True

# Add variance for each feature in the sequence
STD_FEATURES = True

INPUT_MAP = {value: idx for idx, value in enumerate(INPUT_COLUMNS)}
OUTPUT_MAP = {value: idx for idx, value in enumerate(OUTPUT_COLUMNS)}

NUM_INPUTS = len(INPUT_COLUMNS)
NUM_OUTPUTS = len(OUTPUT_COLUMNS)


@plac.annotations(
    input_path=("Path containing the data files to ingest", "option", "P", str),
    input_prefix=("{$prefix}{$year}.npy", "option", "p", str),
    output_path=("Path to saved the trained model", "option", "o", str),
    epochs=("Number of epochs to train", "option", "e", int),
    split=("Chronological train/validation split", "option", "s", float),
    max_depth=("Maximum RF depth", "option", "d", int),
    n_estimators=("Number of trees to create", "option", "n", int),
    n_jobs=("Number of tasks to split creation of forest into", "option", 'j', int)
)
def main(input_path='/project/lindner/moving/summer2018/2019/data-formatted/parallel',
         input_prefix='000_',
         output_path='/project/lindner/moving/summer2018/2019/models/rf.best.pickle',
         epochs=1,
         split=0.66,
         max_depth=None,
         n_estimators=100,
         n_jobs=-1):

    # Load Data
    data_sequences = np.load(os.path.join(input_path, '%ssequences.npy' % input_prefix))
    data_latlong = np.load(os.path.join(input_path, '%slatlong_features.npy' % input_prefix))
    data_sequence_features = np.load(os.path.join(input_path, '%ssequence_features.npy' % input_prefix))
    data_sequences = data_sequences.reshape(data_sequences.shape[0], data_sequences.shape[1] * data_sequences.shape[2])

    labels = np.load(os.path.join(input_path, '%slabels.npy' % input_prefix))

    data = np.concatenate((data_sequences, data_sequence_features), 1)

    # Chronological Train / Validation Split
    split = int(data.shape[0] * split)

    train_X = data[:split]
    train_Y = labels[:split]

    val_X = data[split:]
    val_Y = labels[split:]

    best_r2 = None

    for epoch in range(0, epochs):
        regr = RandomForestRegressor(random_state=epoch, max_depth=max_depth, n_estimators=n_estimators, n_jobs=n_jobs,
                                     verbose=2)
        regr.fit(train_X, train_Y)
        r2 = regr.score(val_X, val_Y)

        save = False

        if best_r2 is None:
            print("epoch(%d) - R^2: %f" % (epoch + 1, r2))
            best_r2 = r2
            save = True
        elif r2 > best_r2:
            print("epoch(%d) - R^2 improved: %f (best: %f)" % (epoch + 1, r2, best_r2))
            best_r2 = r2
            save = True
        else:
            print("epoch(%d) - R^2 did not improve: %f (best: %f)" % (epoch + 1, r2, best_r2))

        if save:
            open(output_path, 'wb').write(pickle.dumps(regr))


if __name__ == '__main__':
    plac.call(main)
