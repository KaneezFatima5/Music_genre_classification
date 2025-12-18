import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.config import *
from torchmetrics.classification import Accuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score

class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=1),
            nn.Tanh(),

            nn.Flatten(),
            nn.Linear(128, 10),
        )

        self.acc = Accuracy(task="multiclass", num_classes=10)
        self.precision = MulticlassPrecision(num_classes=10, average='macro')
        self.recall = MulticlassRecall(num_classes=10, average='macro')
        self.f1 = MulticlassF1Score(num_classes=10, average='macro')

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        preds = logits.argmax(1)
        prec = self.precision(logits, y)
        rec = self.recall(logits, y)
        f1 = self.f1(logits, y)
        acc = self.acc(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_precision", prec)
            self.log(f"{stage}_recall", rec)
            self.log(f"{stage}_f1", f1)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, "test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            # verbose=True,
            min_lr=1e-7
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
