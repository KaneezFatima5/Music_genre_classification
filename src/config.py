import os

CLASS_COLUMN = "class"  # The name of the class column
DROP_COLUMNS = [
    "id",
    "class"
]  # Any columns that should be dropped when training to prevent them from being split on
INDEX_COLUMN = "id"

DPI = 100

ACTIVATION_FUNCTION='sigmoid'
OPTIMIZER='adam'
LOSS_FUNCTION='sparse_categorical_crossentropy'
METRICES='accuracy'

BATCH_SIZE=32
VALIDATION_SPLIT=0.1
HIDDEN_SIZES= [90, 70, 50]
WEIGHT_DECAY=0.001
NUM_WORKERS = 10
UNFREEZE_LAST_N_LAYERS=2  # max 4 layers 

CLASSIFIER = "transfer_learning" #transfer_learning, cnn

DESIRED_SR = 44100
HPSS_MARGIN = 5.0
FRAME_LENGTH = 2048  # analysis window/frame
HOP_LENGTH = 512  # overlap b/w frames
N_MFCC = 13  # number of MFC co-efficients
N_FFT = 2048  # number of samples for each FFT (Fast Fourier Transform)

LR = 0.01 # learning rate
LAMBDA = 0.01  # Regression
EPOCHS = 100  # no. of iteration
ADD_BIAS = True
KERNEL="rbf"


# PATHS

dirname = os.path.dirname(__file__)

# SUBMISSION PATHS

SUB_PATH = os.path.join(
    dirname, "../data/spreadsheets/sub.csv"
)  # The path where the submission dataset for kaggle will be stored

# UNMODIFIED DATASET PATHS

TRAIN_MUSIC_PATH = os.path.join(
    dirname, "../data/music/train"
)  # The path to the training dataset
TEST_MUSIC_PATH = os.path.join(
    dirname, "../data/music/test"
)  # The path to the testing dataset

# PATHS FOR IMAGE BASED MODELS

SPECTROGRAM_PATH = os.path.join(
    dirname, "../data/spectrograms"
)  # The path to where the spectrograms are saved
IMAGE_TRAIN_PATH = os.path.join(
    dirname, "../data/spectrograms/train/layl"
)  # The path to the extracted train data
IMAGE_TEST_PATH = os.path.join(
    dirname, "../data/spectrograms/test/layl"
)  # The path to the extracted train data
MODEL_SAVE_PATH = os.path.join(
    dirname, f"../checkpoints/{CLASSIFIER}.ckpt"
)  # The path to where the model is stored
MODEL_LOAD_PATH = os.path.join(
    dirname, f"../checkpoints/{CLASSIFIER}.ckpt"
)  # The path from where the model is loaded
CHECKPOINTS_PATH = os.path.join(
    dirname, "../checkpoints"
)  # The path to where the model checkpoints are stored


# PATHS FOR CSV BASED MODELS

MODEL_PATH = os.path.join(
    dirname, f"../models/lg.pkl"
)  # The path to where the model is stored to and loaded from
TRAIN_PATH = os.path.join(
    dirname, "../data/spreadsheets/train.csv"
)  # The path to the extracted train data
TEST_PATH = os.path.join(
    dirname, "../data/spreadsheets/test.csv"
)  # The path to the extracted test data

# VISUALIZATION PATHS

VISUALIZATIONS_PATH=os.path.join(
    dirname, "../data/visualizations"
)  # The path to where the visualizations are saved to