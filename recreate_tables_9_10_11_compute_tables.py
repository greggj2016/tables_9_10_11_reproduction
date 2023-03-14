import numpy as np
import pandas as pd
import os
import unlzw3
from pyod.models.knn import KNN
from scipy.stats import norm
from re import split
from pathlib import Path
from umap import UMAP
from copy import deepcopy as COPY
from matplotlib import pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pdb

from recreate_tables_9_10_11_library import T2_thresh_data
from recreate_tables_9_10_11_library import MOD
     
cited_classes = ["normal.", "ftp_write.", "imap.", "multihop.", "nmap.", "phf.", "pod.", "teardrop."]
U2R_attacks = ["loadmodule.", "perl.", "rootkit.", "buffer_overflow."]
all_classes = cited_classes + U2R_attacks

# Importing Dataset 1
kddcup99 = pd.read_csv("D1_corrected.txt", delimiter = ",", header = None)
numeric_cols = np.setdiff1d(kddcup99.columns, [1, 2, 3])
class_rows = kddcup99[41].isin(all_classes)
kddcup99 = kddcup99.loc[class_rows, numeric_cols]
kddcup99[41] = (kddcup99.loc[:, 41] != "normal.").astype(int)
kddcup99 = [kddcup99[np.setdiff1d(kddcup99.columns, [41])], kddcup99[[41]]]

# Importing Dataset 2
wilt1 = pd.read_csv("D2_wilt/training.csv", delimiter = ",")
wilt2 = pd.read_csv("D2_wilt/testing.csv", delimiter = ",")
wilt = pd.concat([wilt1, wilt2])
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
page_blocks = [pd.DataFrame(page_blocks[:, :-1]), pd.DataFrame(page_blocks[:, -1])]

# Importing Dataset 5
Cardio = pd.read_csv("D5_CTG_values.tsv", delimiter = "\t")
Cardio = Cardio.loc[np.isnan(Cardio["NSP"]) == False, :]
Cardio = [Cardio[np.setdiff1d(Cardio.columns, ["NSP"])], Cardio[["NSP"]] > 1]

# Importing Dataset 7
spam = pd.read_csv("D7_spambase.data", delimiter = ",", header = None)
spam = [spam[np.setdiff1d(spam.columns, [57])], spam[[57]]]

# Importing Dataset 8
name = "D8_processed.cleveland.data"
heart = pd.read_csv(name, delimiter = ",", header = None, encoding = "ISO-8859-1")
heart = heart[heart != "?"]
heart = [heart[np.setdiff1d(heart.columns, [13])], heart[[13]] > 0]

# Importing Dataset 9
arr = pd.read_csv("D9_arrhythmia.data", delimiter = ",", header = None)
arr = arr[arr != "?"]
arr = [arr[np.setdiff1d(arr.columns, [279])], arr[[279]] > 1]

# Importing Dataset 10
park = pd.read_csv("D10_parkinsons.data", delimiter = ",")
park = [park[np.setdiff1d(park.columns, ["status", "name"])], park[["status"]]]

N = 1000000
Ncase, Ncon = int(N/10), N - int(N/10)
y_real = np.array(Ncon*[0] + Ncase*[1])
X_controls = norm.ppf(np.linspace(1/Ncon, 1 - (1/Ncon), Ncon))
X_cases = np.array([1, 2, 3, 4, 5, 6, 7, 8]*int(Ncase/8))
X = np.concatenate([X_controls, X_cases])

datasets = [kddcup99, wilt, page_blocks, Cardio, spam, heart, arr, park]
names = ["kddcup99", "wilt", "page_blocks", "Cardio", "spam", "heart", "arr", "park"]
best_k_MOD, best_k_KNN = [], []
for ds, name in zip(datasets, names):

    MOD_ROC_AUCs, KNN_ROC_AUCs = [], []
    void, outliers = ds[0].to_numpy(), ds[1].to_numpy().reshape(-1)
    for k in range(2, 101):
        fname1 = "MOD_scores/" + name + "_MOD_" + str(k) + "k.txt"
        fname2 = "KNN_scores/" + name + "_KNN_" + str(k) + "k.txt"
        MOD_scores = pd.read_csv(fname1, delimiter = "\t", header = None).to_numpy().reshape(-1)
        KNN_scores = pd.read_csv(fname2, delimiter = "\t", header = None).to_numpy().reshape(-1)
        MOD_ROC_AUCs.append(roc_auc_score(outliers, MOD_scores))
        KNN_ROC_AUCs.append(roc_auc_score(outliers, KNN_scores))

    best_k_MOD.append(np.argmax(MOD_ROC_AUCs) + 2)
    best_k_KNN.append(np.argmax(KNN_ROC_AUCs) + 2)

best_k_vals = pd.DataFrame(np.array([names, best_k_MOD, best_k_KNN]).T)
best_k_vals.columns = ["dataset", "MOD best k", "KNN best k"]
best_k_vals.to_csv("best_k_vals.txt", sep = "\t", header = True, index = False)
