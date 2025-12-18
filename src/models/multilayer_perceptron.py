import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils import *
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def data_loader(dataset, batch_size, is_shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        """Using neural network module from pytorch"""
        self.layers=nn.ModuleList()
        """Size of input layer == No. of features"""
        in_size=input_size

        """Dynamic number and size of hidden layers - controlled from config file as a hyperparameter"""
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_size, hidden_size))
            in_size=hidden_size

        """Output size """
        self.out=nn.Linear(in_size, output_size)
        """Activation function"""
        self.relu= nn.ReLU()
        """dropout certain neurons to implement regularization / prevent overfitting"""
        self.dropout = nn.Dropout(dropout_rate)
        """Loss Function """
        self.cost= nn.CrossEntropyLoss()

    def forward(self, x):
        """"Forward calculation impl layer wise"""
        for layer in self.layers:
            x=self.relu(layer(x))
            x=self.dropout(x)
        x=self.out(x)
        return x
    
class multilayer_perceptron:
    def __init__(self,train_df, batch_size, lr, weight_decay, hidden_size, val_split):
        self.batch_size=batch_size
        self.lr=lr
        self.weight_decay=weight_decay
        """Standard scalar for standardization"""
        self.scalar=StandardScaler()
        """Using label encoder for encoding the classes"""
        self.le=LabelEncoder()
        
        x_train=get_features(train_df)
        y_train=get_class_col(train_df)

        """transform classes to encoded form"""
        y_train=self.le.fit_transform(y_train)
        x_val=None
        y_val=None
        """train - validation split"""
        if val_split>0:
            x_train, x_val, y_train, y_val= train_test_split(
                x_train, y_train, test_size=val_split, random_state=42, stratify=y_train
            )

        """Standardize train data"""
        x_train=self.scalar.fit_transform(x_train)
        """Standardize validation data"""
        x_val=self.scalar.transform(x_val) if x_val is not None else None

        """Convert datasets into tensors"""
        self.train_dataset=TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ) 
        self.val_dataset = None
        if x_val is not None:
            self.val_dataset = TensorDataset(
                torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )

        """Define input size i.e. get number of features"""
        input_size=x_train.shape[1]
        """Output size == number of classes"""
        output_size=len(np.unique(y_train))
        """Load model"""
        self.model=MLP(input_size, hidden_size, output_size)
        """Using Adam optimizer - can control LR and weight decay from config file"""
        self.optimizer=optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        """create data loaders for train and validation dataset"""
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size) if self.val_dataset else None
        self.losses=[]

    def train(self, epochs):

        """Trian model on defined number of epoch"""
        for epoch in range(epochs):
            self.model.train()
            train_loss=0.0
            correct_train=0
            total_train=0
            y_true = []
            y_pred = []

            """For each epoch do the calculation"""
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss=self.model.cost(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss+=loss.item()

                """Get the predicted values """
                _, predicted=torch.max(outputs, 1)
                """append predictions and target values """
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

                """Calculate correct values"""
                total_train+=targets.size(0)
                correct_train+= (predicted==targets).sum().item()
            """Calculate loss, train accuracy, recall and precision"""
            avg_train_loss = train_loss/len(self.train_loader)
            train_accuracy=correct_train/total_train
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(self.train_loader):.4f}, Train Accuracy: {100 * train_accuracy:.2f}%, Precision: {precision:.4f}, recall: {recall:.4f}, F1-Score: {f1:.4f} ")

            self.losses.append(avg_train_loss)

    def evaluate_validation(self):
        """Evaluate validation dataset"""
        if self.val_loader is None:
            raise RuntimeError("No validation dataset available.")
        self.model.eval()
        val_loss=0.0
        corrected_val=0
        total_val=0

        y_true = []
        y_pred = []

        """Same process to evaluation validation dataset, calculate accuracy and precision"""
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss=self.model.cost(outputs, targets)
                val_loss+=loss.item()

                _, predicted = torch.max(outputs, 1)

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

                total_val += targets.size(0)
                corrected_val += (predicted==targets).sum().item()

        val_accuracy=corrected_val/total_val
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch {1}/{2}, Validation Loss: {val_loss / len(self.val_loader):.4f}, Validation Accuracy: {100 * val_accuracy:.2f}%, Precision: {precision:.4f}, recall: {recall:.4f}, F1-Score: {f1:.4f} ")


    def evaluate(self, test_df):
        """Evaluate test dataset"""
        x_test=get_features(test_df)
        """Standardize test dataset on same scalar as was used for train data standardization"""
        x_test=self.scalar.transform(x_test)

        """Convert datset to tensors"""
        inputs=torch.tensor(x_test, dtype=torch.float32)
        self.model.eval()
        """Calculate probabilities using softmax, get argmax"""
        with torch.no_grad():
            logits=self.model(inputs)
            probs=torch.softmax(logits, dim=1)
            preds=torch.argmax(probs, dim=1).numpy()

        """Convert encoded classes to class labels i.e. genre names"""
        pred_labels= self.le.inverse_transform(preds)
        y_id=get_index_col(test_df)
        """retrun dataframe of prediction genres"""
        return pd.DataFrame({"id":y_id, "class": pred_labels})

