from pytorch_lightning.loggers import TensorBoardLogger
from config import *
from src.models.cnn import CNN
from pathlib import Path
import pytorch_lightning as pl
from data_modules.music_data_module import MusicDataModule
from src.models.transfer_learning import TransferLearning
import torch
import time

def image_train(dm, checkpoints_path, save_path, classifier):

    # Initialize the model
    if classifier == "cnn":
        model = CNN()
    elif classifier == "transfer_learning":
        model=TransferLearning(lr=LR, num_classes=len(dm.classes), unfreeze_last_n_layers=UNFREEZE_LAST_N_LAYERS)
    else:
        raise ValueError('Invalid classifier was entered.')

    torch.set_float32_matmul_precision('high')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        dirpath=checkpoints_path,
        filename='best_model_{epoch:02d}_{val_loss:.2f}'
    )

    # Early stopping - stop if validation loss doesn't improve
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Wait 10 epochs before stopping
        mode='min',
        verbose=True
    )
    
    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    logger = TensorBoardLogger('tb_logs', name=classifier)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, callbacks=[checkpoint_callback, early_stop_callback, lr_monitor])

    start_time = time.time()

    # Train the model
    trainer.fit(model, dm)

    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds")

    trainer.save_checkpoint(save_path)

if __name__ == "__main__":
    data_module = MusicDataModule(IMAGE_TRAIN_PATH, IMAGE_TEST_PATH, BATCH_SIZE, NUM_WORKERS, VALIDATION_SPLIT, CLASSIFIER)
    data_module.setup()
    checkpoints_loc = Path(CHECKPOINTS_PATH)
    save_loc = Path(MODEL_SAVE_PATH)
    image_train(data_module, checkpoints_loc, save_loc, CLASSIFIER)