import matplotlib.pyplot as plt
from src.config import *
from sklearn.decomposition import PCA
import numpy as np
from src.logistic_regression import standardize
import matplotlib as mpl

def plot_training_loss(model, folder_path):
    plt.plot(model.losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(folder_path+"/training_loss.png")
    plt.close()


def analyze_class_samples(train_df, class_name):
    if class_name == "all":
        class_df=train_df
    else:
        class_df=train_df[train_df[CLASS_COLUMN] == class_name]
    x_df=class_df.drop(DROP_COLUMNS, axis=1)
    x_df = standardize(x_df, x_df)
    indexes = class_df[INDEX_COLUMN]
    classes = class_df[CLASS_COLUMN]
    pca = PCA(n_components=3)
    x_df = x_df.transpose()
    pca.fit(x_df)
    components = pca.components_
    evr = pca.explained_variance_ratio_
    print(evr)
    print(evr.sum())


    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='3d')
    unique_classes = np.unique(classes)
    colors = mpl.colormaps["tab10"]
    print(components.shape)
    for i, label in enumerate(unique_classes):
        print(label)
        ax.scatter(components[0,classes==label],
                    components[1,classes==label],
                    components[2,classes==label],
                    color=colors(i),
                    label=label)
    if class_name != "all":
        for i, txt in enumerate(indexes):
            ax.text(components[0][i], components[1][i], components[2][i], txt, size="small")
    plt.legend()
    plt.show()


