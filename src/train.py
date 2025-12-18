#!/usr/bin/env python3

from pathlib import Path
import time

from utils import *
from config import *
from src.models import multilayer_perceptron

def train(train_path, save_path):
    train_df = load_spreadsheet(train_path)
    print("Training model")
    start_time = time.time()
    print("Using Multi layer perceptron")
    model = multilayer_perceptron.multilayer_perceptron(
        train_df,
        BATCH_SIZE,
        LR, WEIGHT_DECAY, HIDDEN_SIZES, VALIDATION_SPLIT
    )
    model.train(EPOCHS)
    model.evaluate_validation()
    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds")

    print("Saving model")
    save_model(save_path, model)


if __name__ == "__main__":
    train_loc = Path(TRAIN_PATH)
    save_loc = Path(MODEL_PATH)
    train(train_loc, save_loc)
