from config import *
from pathlib import Path
from utils import *
from visualizations import *

def visualize(model_path, train_path, test_path):
    train_df = load_spreadsheet(train_path)

    analyze_class_samples(train_df, "classical")






if __name__ == "__main__":
    model_loc = Path(MODEL_PATH)
    train_loc = Path(TRAIN_PATH)
    test_loc = Path(TEST_PATH)
    sub_loc = Path(SUB_PATH)
    visualize(model_loc, train_loc, test_loc)