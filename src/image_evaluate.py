import pandas as pd
from config import *
from src.models.cnn import CNN
from src.models.transfer_learning import TransferLearning
from pathlib import Path
from data_modules.music_data_module import MusicDataModule
import torch
import time
from utils import write_spreadsheet

def image_evaluate(dm, model_path, sub_path, classifier):
    if classifier == "cnn":
        model = CNN.load_from_checkpoint(model_path)
    elif classifier=="transfer_learning":
        model=TransferLearning.load_from_checkpoint(model_path)
    else:
        raise ValueError('Invalid classifier was entered.')

    model.eval()

    test_dataloader = dm.test_dataloader()

    sample_names = []
    predictions = []

    start_time = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader, 0):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            sample_name, _ = test_dataloader.dataset.samples[i]
            predictions.append(dm.classes[predicted.item()])
            sample_names.append(Path(sample_name).stem)

    end_time = time.time()
    print(f"Evaluation took: {end_time - start_time} seconds")

    df = pd.DataFrame({INDEX_COLUMN: sample_names, CLASS_COLUMN: predictions})
    write_spreadsheet(df, sub_path)

if __name__ == "__main__":
    data_module = MusicDataModule(IMAGE_TRAIN_PATH, IMAGE_TEST_PATH, BATCH_SIZE, NUM_WORKERS, VALIDATION_SPLIT, CLASSIFIER)
    data_module.setup()
    model_loc = Path(MODEL_LOAD_PATH)
    sub_loc = Path(SUB_PATH)
    image_evaluate(data_module, model_loc, sub_loc, CLASSIFIER)