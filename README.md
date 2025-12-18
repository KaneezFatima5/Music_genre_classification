# CS529-Neural Network 
### Repo Structure
The repository is organized into 5 main folders:
* checkpoints - Contains current best model from NN and Transfer learning.
* data - Contains all the datasets and submission data.
* models - Contains the current best model, and is where trained models are stored using pickle.
* report - Contains the tex files for the report.
* src - Contains source files for the project, also contains config.py.

## File Manifest

### ./models
* best_model_epoch=x_val_loss=y.ckpt - The current best model, saved using pytorch checkpoints.

### ./data
This folder may be empty as the dataset must be removed when submitting.
* music folder - Contains test and train folders containing audio music files respectively 
* spreadsheets folder - Contains following 
    * test.csv - The test dataset containing all extracted features.
    * train.csv - The train dataset containing all extracted features.
    * sub.csv - The file storing the submission to kaggle.
* spectrograms folder - Contains following 
    * test - Contains spectrograms for test dataset.
    * train - Contains spectrograms for train dataset.
* list_test.csv - ids for test dataset.
* potential_outliers.txt - Analyzed data of outliers.

### ./models
* selected.pkl - The current best model, saved using pickle trained on MLP model.

### ./report
* images - images for report
* report.pdf - The compiled report.
* report.tex - The uncompiled tex file of the report.
* /report_archive - Old report format that we planned on using before swapping over to a simpler one due to it not being part of the rubric.

### ./src
* data_modules 
    * __init__.py 
    * music_data_module.py - Contains code for spectrograms transformations, train-validation split and dataloader formation
* models 
    * __init__.py 
    * cnn.py - Convolutional Neural Network training code 
    * multilayer_perceptron.py - Training and evaluation code for multi-layer perceptron model using pytorch
    * transfer_learning.py - Utilized pretrained mode - ResNet50 for transfer learning 
* config.py - Contains all configurable hyperparameters.
* create_spectrogram.py - Generating soectrogram image for music files
* data_analysis.py - Generating heat maps, and t_SNE clustering maps for data visualization
* evaluate.py - Uses the model generated using MPLs stored at the given path in config.py, evaluates all the test data and stores the output in the path given by config.py
* extract.py - extract various features from the audio files.
* image_evaluate.py - Model evaluation on test dataset for models trained using spectrograms i.e. NN and transfer learning 
* image_train.py - Training for models trained using spectrograms i.e. NN and transfer learning 
* main.py - Simply calls the functions from evaluate.py and train.py, while passing configuration information from config.py
* train.py - Trains a models (MLPs) using configuration information from config.py, saves the model at the path given by config.py
* utils.py - Contains some utility functions such as load spreadsheets, get feature matrix, get class matrix etc.
* visualization.py - Visualizing the misclassified samples.
* visualize.py - Simply calls the functions from visualization.py.


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