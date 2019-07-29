# Average window_stride elements together to form a single row
WINDOW_STRIDE = 12

SAMPLE_HOURS = WINDOW_STRIDE / 12.0

# Number of future samples to mean for prediction
PREDICTION_WINDOW = int(24 / SAMPLE_HOURS)

# Length of the windowed sequence
# orig
SEQUENCE_LENGTH = int(7*24 / SAMPLE_HOURS)
# testing
#SEQUENCE_LENGTH = 2

# Input Features
#INPUT_COLUMNS = ['epoch', 'day_of_year', 'hour', 'temp', 'windspd', 'winddir', 'wind_x_dir', 'wind_y_dir', 'no', 'no2', 'nox', 'o3']
INPUT_COLUMNS = ['Latitude', "Longitude", 'hour', 'day_of_year', 'co', 'humid', 'nox', 'pm25','so2', 'solar', 'temp', 'wind_x_dir', 'wind_y_dir', 'windspd', 'dew', 'o3']

#OUTPUT_COLUMNS = ['no', 'no2', 'nox', 'o3']
OUTPUT_COLUMNS = ['o3']

# Take the FFT of each sqeuence and use as features
FFT_FEATURES = False

# Fit the sequence to y = mx+b and add the coeff / intercept
REGRESSION_FEATURES = True

# Add variance for each feature in the sequence
STD_FEATURES = True

INPUT_MAP = {value: idx for idx, value in enumerate(INPUT_COLUMNS)}
OUTPUT_MAP = {value: idx for idx, value in enumerate(OUTPUT_COLUMNS)}

#We don't have INPUT_MAP_['temp'] key
ENRICH_START = INPUT_MAP['temp']

NUM_INPUTS = len(INPUT_COLUMNS)
NUM_OUTPUTS = len(OUTPUT_COLUMNS)

# All the timesteps features unrolled length
UNROLLED_SEQUENCE_LENGTH = NUM_INPUTS * SEQUENCE_LENGTH

SITES = {
'48_201_0695': {'': 450, 'Site ID': '48_201_0695', 'Site Name': 'UH Moody Tower', 'Longitude': -95.3414,
                         'Latitude': 29.7176, 'Region': 12, 'Activation Date': '31-Mar-10', 'Deactivation Date': ''},

'48_201_0416': {'': 427, 'Site ID': '48_201_0416', 'Site Name': 'Park Place', 'Longitude': -95.294722,
         'Latitude': 29.686389, 'Region': 12, 'Activation Date': '22-Feb-06', 'Deactivation Date': ''}
}