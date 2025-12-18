#!/usr/bin/env python3

from pathlib import Path
from train import train
from evaluate import evaluate
from config import *


def main():
    train_path = Path(TRAIN_PATH)
    test_path = Path(TEST_PATH)
    model_path = Path(MODEL_PATH)
    sub_path = Path(SUB_PATH)

    train(train_path, model_path)
    evaluate(model_path, test_path, sub_path)


if __name__ == "__main__":
    main()
