import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as func
from torchvision.models import resnet50, ResNet50_Weights
from src.config import *
from src.utils import *
from pathlib import Path
from torchmetrics.classification import Accuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score


class TransferLearning(pl.LightningModule):
    def __init__(self, lr=1e-3, num_classes=10, unfreeze_last_n_layers=2):
        super().__init__()
        self.save_hyperparameters()

        """Load model with default weights"""
        self.model= resnet50(weights = ResNet50_Weights.DEFAULT)

        """Freeze layers"""
        for param in self.model.parameters():
            param.requires_grad = False

        """Un freeze last n layers"""
        layers_to_unfreeze= []
        if unfreeze_last_n_layers>=1:
            layers_to_unfreeze.append(self.model.layer4)
        if unfreeze_last_n_layers>=2:
            layers_to_unfreeze.append(self.model.layer3)
        if unfreeze_last_n_layers>=3:
            layers_to_unfreeze.append(self.model.layer2)
        if unfreeze_last_n_layers>=4:
            layers_to_unfreeze.append(self.model.layer1)
        
        """Save a identifier for unforzen layers"""
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad=True

        """Replace Classfier head i.e. change the length of f.c (fully connected/output layer) and other layers"""
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),  #dropout for regularization
            nn.Linear(num_features, num_classes)
        )

        """Get accuracy, precision, recall and F1 score"""
        self.acc = Accuracy(task="multiclass", num_classes=10)
        self.precision = MulticlassPrecision(num_classes=10, average='macro')
        self.recall = MulticlassRecall(num_classes=10, average='macro')
        self.f1 = MulticlassF1Score(num_classes=10, average='macro')

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        """Using cross entropy as loss function"""
        loss = nn.CrossEntropyLoss()(logits, y)
        """Get arg max on predicted values"""
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
        """Train models to loss and accuracy"""
        loss, acc = self.shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validate model"""
        loss, acc = self.shared_step(batch, "val")
        return loss

    def configure_optimizers(self):
        #separate pretrained and new parameters
        pretrained_params = []
        new_params =[]

        #collect unfrozen pretrained parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'fc' not in name:
                pretrained_params.append(param)
            elif param.requires_grad and 'fc' in name:
                new_params.append(param)

        #Using different learning rate for different layers
        optimizer= torch.optim.Adam([
            {'params': pretrained_params, 'lr': self.hparams.lr*0.1},
            {'params': new_params, 'lr': self.hparams.lr}
        ], weight_decay=WEIGHT_DECAY)

        #learning rate scheduler
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
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
    # def evaluate(self, data_module):
    #     self.model.eval()
    #     predictions=[]
    #     sample_names=[]
    #     test_loader=data_module.test_dataloader()

    #     with torch.no_grad():
    #         for x, (images, _) in enumerate(test_loader):
    #                 images = images.to(self.device)
    #                 outputs = self.model(images)
    #                 pred = torch.argmax(outputs, 1).item()
    #                 sample_name, _ = test_loader.dataset.samples[x]
    #                 predictions.append(data_module.classes[pred])
    #                 sample_names.append(Path(sample_name).stem)
    #         return pd.DataFrame({"id":sample_names, "class":predictions})

            
            

# if __name__=="__main__":
#     train_df=load_spreadsheet(Path(TRAIN_PATH))

#     model=transfer_learning(train_df)
#     model.train(EPOCHS)
#     model.evaluate_validation()
#     sub_loc = Path(SUB_PATH)
#     model.evaluate(sub_loc)