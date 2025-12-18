from config import *
import pandas as pd
import pickle


def load_spreadsheet(path):
    return pd.read_csv(path)

def get_features(dataset):
    return dataset.drop(columns=DROP_COLUMNS)

def get_index_col(dataset):
    return dataset[INDEX_COLUMN]

def get_class_col(dataset):
    return dataset[CLASS_COLUMN]

def write_spreadsheet(df, path):
    df.to_csv(path, index=False)


def save_model(save_path, model):
    with open(save_path, "wb") as file:
        pickle.dump(model, file)


def load_model(load_path):
    with open(load_path, "rb") as file:
        return pickle.load(file)

