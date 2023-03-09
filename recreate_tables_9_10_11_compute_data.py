import numpy as np
import pandas as pd
import os
import unlzw3
import argparse
from pyod.models.knn import KNN
from re import split
from pathlib import Path
from umap import UMAP
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from itertools import combinations
from tqdm import tqdm
import pdb

from recreate_tables_9_10_11_library import MOD

parser = argparse.ArgumentParser()
parser.add_argument('--k ', type = int, action = "store", dest = "k")
k = parser.parse_args().k
     
cited_classes = ["normal.", "ftp_write.", "imap.", "multihop.", "nmap.", "phf.", "pod.", "teardrop."]
U2R_attacks = ["loadmodule.", "perl.", "rootkit.", "buffer_overflow."]
all_classes = cited_classes + U2R_attacks

# Importing Dataset 1
kddcup99 = pd.read_csv("D1_corrected.txt", delimiter = ",", header = None)
numeric_cols = np.setdiff1d(kddcup99.columns, [1, 2, 3])
class_rows = kddcup99[41].isin(all_classes)
kddcup99 = kddcup99.loc[class_rows, numeric_cols]
kddcup99[41] = (kddcup99.loc[:, 41] != "normal.").astype(int)
kddcup99.to_csv("data/kddcup99.txt", sep = "\t", header = True, index = False)
kddcup99 = [kddcup99[np.setdiff1d(kddcup99.columns, [41])], kddcup99[[41]]]

# Importing Dataset 2
wilt1 = pd.read_csv("D2_wilt/training.csv", delimiter = ",")
wilt2 = pd.read_csv("D2_wilt/testing.csv", delimiter = ",")
wilt = pd.concat([wilt1, wilt2])
wilt.to_csv("data/wilt.txt", sep = "\t", header = True, index = False)
wilt = [wilt[np.setdiff1d(wilt.columns, ["class"])], wilt[["class"]] == "w"]

# Ignoring Dataset 3 due to heavy modification

# Importing Dataset 4
compressed_page_blocks = open("D4_page-blocks.data.Z", 'rb').read()
page_blocks = str(unlzw3.unlzw(compressed_page_blocks))
page_blocks = page_blocks.split("\\n")
page_blocks = [split("\s+", row)[1:] for row in page_blocks]
page_blocks.pop()
page_blocks = np.array(page_blocks).astype(float)
page_blocks[:, -1] = (page_blocks[:, -1] != 1).astype(int)
pd.DataFrame(page_blocks).to_csv("data/page_blocks.txt", sep = "\t", header = True, index = False)
page_blocks = [pd.DataFrame(page_blocks[:, :-1]), pd.DataFrame(page_blocks[:, -1])]

# Importing Dataset 5
cardio = pd.read_csv("D5_CTG_values.tsv", delimiter = "\t")
cardio = cardio.loc[np.isnan(cardio["NSP"]) == False, :]
cardio.to_csv("data/cardio.txt", sep = "\t", header = True, index = False)
cardio = [cardio[np.setdiff1d(cardio.columns, ["NSP"])], cardio[["NSP"]] > 1]

# Importing Dataset 7
spam = pd.read_csv("D7_spambase.data", delimiter = ",", header = None)
spam.to_csv("data/spam.txt", sep = "\t", header = True, index = False)
spam = [spam[np.setdiff1d(spam.columns, [57])], spam[[57]]]

# Importing Dataset 8
name = "D8_processed.cleveland.data"
heart = pd.read_csv(name, delimiter = ",", header = None, encoding = "ISO-8859-1")
heart = heart[heart != "?"]
heart.to_csv("data/heart.txt", sep = "\t", header = True, index = False)
heart = [heart[np.setdiff1d(heart.columns, [13])], heart[[13]] > 0]

# Importing Dataset 9
arr = pd.read_csv("D9_arrhythmia.data", delimiter = ",", header = None)
arr = arr[arr != "?"]
arr.to_csv("data/arr.txt", sep = "\t", header = True, index = False)
arr = [arr[np.setdiff1d(arr.columns, [279])], arr[[279]] > 1]

# Importing Dataset 10
park = pd.read_csv("D10_parkinsons.data", delimiter = ",")
park.to_csv("data/park.txt", sep = "\t", header = True, index = False)
park = [park[np.setdiff1d(park.columns, ["status", "name"])], park[["status"]]]

if not os.path.exists("MOD_scores"):
    os.mkdir("MOD_scores")
datasets = [kddcup99, wilt, page_blocks, cardio, spam, heart, arr, park]
names = ["kddcup99", "wilt", "page_blocks", "cardio", "spam", "heart", "arr", "park"]
 
for ds, name in zip(datasets, names):
    
    X, y = ds[0].to_numpy(), ds[1].to_numpy().reshape(-1)
    X = X.astype(float)
    X_std = np.std(X, axis = 0)
    X = X[:, X_std > 0]
    for i in range(len(X[0])): X[np.isnan(X[:, i]), i] = np.nanmean(X[:, i])
    fname = "MOD_scores/" + name + "_MOD_" + str(k) + "k.txt"
    MOD_scores = MOD(X, k, 3)
    MOD_df = pd.DataFrame(MOD_scores)
    MOD_df.to_csv(fname, sep = "\t", header = False, index = False)

    fname = "KNN_scores/" + name + "_KNN_" + str(k) + "k.txt"
    model = KNN(method = 'largest', metric = 'euclidean', n_neighbors = k)
    fitted_model = model.fit(X)
    KNN_scores = fitted_model.decision_scores_
    KNN_df = pd.DataFrame(MOD_scores)
    KNN_df.to_csv(fname, sep = "\t", header = False, index = False)

