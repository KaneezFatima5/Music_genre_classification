#!/usr/bin/env python3

from pathlib import Path
from utils import *
from config import *


def evaluate(model_path, test_path, sub_path):
    """Evaluate a spreadsheet of test data, given the path to the model, the test data, and the location where the evaluations will be saved."""

    print("Loading test data")
    test_df = load_spreadsheet(test_path)

    print("Loading model")
    model = load_model(model_path)

    print("Evaluating test data")
    sub_df = model.evaluate(test_df)

    print("Writing submission")
    write_spreadsheet(sub_df, sub_path)


if __name__ == "__main__":
    model_loc = Path(MODEL_PATH)
    test_loc = Path(TEST_PATH)
    sub_loc = Path(SUB_PATH)
    evaluate(model_loc, test_loc, sub_loc)
