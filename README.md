🎶 Music Genre Classification with Deep Learning & Transfer Learning
===================================================================
From spectrograms → CNNs → ResNet-50 → scalable deep learning

🚀 Overview
----------------
This project explores deep learning approaches for music genre classification, progressing from MLPs to CNNs, and finally transfer learning using ResNet-50.

The focus is on representation learning, scalability, and performance comparison across architectures.

🎯 Objective
-------------
Predict the genre of a music track by learning patterns directly from spectrogram representations using deep neural networks.

🔄 End-to-End Pipeline
Audio Files
   ↓
Feature Extraction
   ↓
Spectrogram Generation
   ↓
MLP / CNN Training
   ↓
Transfer Learning (ResNet-50)
   ↓
Evaluation & Comparison

### 1\. Feature Extraction

Audio files were converted into structured numerical representations using:

-   MFCCs (Mel-Frequency Cepstral Coefficients)
-   Chroma features
-   Spectral Centroid
-   Spectral Bandwidth
-   Zero-Crossing Rate
-   Tempo-related features

These features capture both frequency and temporal characteristics of music.

### 2\.  Spectrogram Generation

-   Converted audio signals into time–frequency spectrograms
-   Treated spectrograms as images
-   Enabled CNNs to learn:
-   Frequency textures
-   Temporal rhythm patterns
-   Harmonic structures

### 3\. Model Architectures
🔹 Multi-Layer Perceptron (MLP)

-   Trained on structured spectrogram features 
-   Hyperparameter-controlled:
    -   Number of layers
    -   Layer size 

🔹 Convolutional Neural Network (CNN)

-   Trained directly on spectrogram images
-   Learned spatial time–frequency patterns
-   Outperformed MLPs due to automatic feature learning

🔹 Transfer Learning — ResNet-50

-   Pretrained on ImageNet
-   Replaced final classification layer
-   Selective layer unfreezing via hyperparameters
-   Compared:
|   -   Fully frozen backbone
|   -   Partially fine-tuned models
Demonstrated effective cross-domain knowledge transfer.

### \. Evaluation Metrics

-   Accuracy
-   Precision / Recall / F1-score
-   Confusion Matrix
-   Training & validation loss curves

🔍 Key Insights
--------------------
-   CNNs outperform MLPs on spectrogram data
-   Transfer learning improves convergence and generalization
-   Partial unfreezing balances performance and overfitting
-   Pretrained vision models transfer well to audio tasks

🛠️ Tech Stack
--------------
-   Python
-   NumPy
-   Scikit-learn
-   PyTorch
-   PyTorch Lightning
-   Torchvision
-   Librosa
-   Matplotlib / Seaborn

📁 Repository Structure 
### Repo Structure
The repository is organized into 5 main folders:
-    checkpoints - Contains current best model from NN and Transfer learning.
-    data - Contains all the datasets and submission data.
-    models - Contains the current best model, and is where trained models are stored using pickle.
-    report - Contains the tex files for the report.
-    src - Contains source files for the project, also contains config.py.

## Running Code
There exist 3 different options for running this program, which are all detailed below.

### Common Steps
First, verify that the paths in `./src/config.py` are all valid, and that the hyperparameters are set to what you desire. 

You must be in the src folder when running anything.

### Training and Evaluating model.
To train and evaluate a model, simply run main.py after doing the common steps. The model will be saved at the location specified. You will still need to upload the submission dataset to kaggle manually.
```
python3 main.py
```

### Training a model.
### a. MPLs.
To simply train a model, run train.py after doing the common steps. The model will be saved at the location specified.

### b. NN and Transfer Learning
To train model on neural netwrok or using transfer learning, simply run image_train.py file to train model and the model will be saved at specified location.

### Evaluating a model.
### a. MPLs.
To evaluate a model, run evaluate.py after doing the common steps. You will still need to upload the submission dataset to kaggle manually.

```
python3 evaluate.py
```
### b. NN and Transfer Learning
To evaluate model on neural netwrok or using transfer learning, simply run image_evaluate.py file and the sumission file will be saved on given format at specified location for kaggle submission.

## Contributions
* Sathvik Quadros - Wrote code for creating spectrograms, NN model implementation and worked on the report.
* Fatima - Wrote code for multi-layer perceptron model implementation and transfer learning. Created readme file

## Kaggle Score, Accuracy, and Date Run
* Kaggle Score - 0.74
* Average accuracy on validation set - 0.68
* Date run - December 9th, 2025