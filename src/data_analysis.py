from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import seaborn as sns
from utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE



import numpy as np

train_path=Path(TRAIN_PATH)
df =  load_spreadsheet(train_path)  # path to your file
df = df.drop(columns=["id"]) 
df = df.drop(columns=["Tempo"]) 

"""Heat Map"""
plt.figure(figsize=(22, 18))
corr = df.drop(columns=["class"]).corr()

sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
# plt.show()

"""Find Redundant Features"""
corr_matrix = corr.abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

redundant_features = [
    column for column in upper_tri.columns 
    if any(upper_tri[column] > 0.90)
]

print("Highly correlated (redundant) features:")
for f in redundant_features:
    print(f)


"""Random forest Importance of features"""

# split features + target
X = df.drop(columns=["class"])
y = df["class"]

# encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# train RF model
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y_enc)

# extract importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print(importances)



"""t_SNE clustering visualization"""
X = df.drop(columns=["class"])   # features
y = df["class"]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tsne = TSNE(
    n_components=2,
    perplexity=30,        # controls “cluster tightness”
    learning_rate=200,
    init='pca',
    random_state=42
)

tsne_result = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y.astype('category').cat.codes)
plt.title("t-SNE Visualization (2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()


tsne3 = TSNE(n_components=3, perplexity=30, random_state=42)
tsne_3d = tsne3.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2],
           c=y.astype('category').cat.codes)

ax.set_title("t-SNE Visualization (3D)")
plt.show()


# corr_matrix = X.corr().abs()

# # correlation threshold
# threshold = 0.85

# # set to store features to drop
# to_drop = set()

# # iterate through correlations
# for col in corr_matrix.columns:
#     for row in corr_matrix.index:
#         if col != row and corr_matrix.loc[row, col] > threshold:
#             # drop less important feature
#             if importances.set_index('feature').loc[row].importance > \
#                importances.set_index('feature').loc[col].importance:
#                 to_drop.add(col)
#             else:
#                 to_drop.add(row)

# print("Drop these features:")
# print(to_drop)