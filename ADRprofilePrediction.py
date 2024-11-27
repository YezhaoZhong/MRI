#!/usr/bin/env python

import pandas as pd
import networkx as nx
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_auc_score
import random
from sklearn import metrics
import time
from scipy.linalg import inv
from joblib import Parallel, delayed
from sklearn.utils import parallel_backend
from itertools import product
import warnings
from Models import runModels, setDist
from collections import defaultdict

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


random.seed(1949)
np.random.seed(1949)


def option(str1, str2, str3):
    global methodOption
    global tuningMetrice
    global validationOption

    methodOption = str1
    tuningMetrice = str2
    validationOption = str3


def Pairs2Mat(path,colname1,colname2,sep = "\t"):
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.read_csv(path,sep=sep,low_memory=False)
    df_nadroped = df[[colname1,colname2]].dropna()
    drug_names = df_nadroped[colname1] # put col of df in var
    drug_names_upper = [drug_name.upper() for drug_name in drug_names]
    features = df_nadroped[colname2]
    features_upper = [feature.upper() for feature in features]

    edgelist = zip(features_upper, drug_names_upper)
    ##making Biparite Graph##
    B = nx.DiGraph()
    B.add_nodes_from(features_upper,bipartite = 0)
    B.add_nodes_from(drug_names_upper,bipartite = 1)
    B.add_edges_from(edgelist)
    drug_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}
    feature_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
    drug_nodes = list(drug_nodes)
    drug_nodes.sort()
    feature_nodes = list(feature_nodes)
    feature_nodes.sort()
    ###Getting the Bi-Adjacency matrix between side effects and drugs ###################
    matrix_all = biadjacency_matrix(B, row_order = feature_nodes, column_order = drug_nodes) # create biadjacency matrix for drug side effect graph
    # drug_nodes_upper = [drug_node.upper() for drug_node in drug_nodes]
    matrix_all = pd.DataFrame(matrix_all.A.T, columns=feature_nodes, index=drug_nodes)
    
    return matrix_all # drug*feature

def FeaturePreprocess(df_all, drug):
    
    drug_nodes_df = np.intersect1d(df_all.index, drug)
    df = df_all.loc[drug_nodes_df]
    _, q = df.shape
    drug_nodes_diff = np.setdiff1d(drug, (df.index).tolist())
    n = len(drug_nodes_diff)
    df_diff = pd.DataFrame(np.zeros((n, q)))
    df_diff.index = drug_nodes_diff
    df_diff.columns = df.columns
    df_all = pd.concat([df, df_diff], axis = 0)
    featureMat = df_all.loc[drug]
    return np.array(featureMat)



def adjFmax(y, y_pred, ic):
    
    _, n = y.shape
    thresholdList = np.arange(0, 1.001, 0.001).tolist()
    N_thresholdList = len(thresholdList)
    n_e = np.zeros(N_thresholdList)
    icI_FN = np.zeros(N_thresholdList)
    icI_FP = np.zeros(N_thresholdList)
    icI_TP = np.zeros(N_thresholdList)

    TP = np.zeros(N_thresholdList)
    TN = np.zeros(N_thresholdList)
    FP = np.zeros(N_thresholdList)
    FN = np.zeros(N_thresholdList)

    Obs = y.copy()
    for i in range(N_thresholdList):
        Pred = (y_pred >= thresholdList[i])
        n_e[i] = sum((Pred.sum(axis = 0)) > 0)
        Result = Pred - Obs
        Result2 = Pred + Obs
        FP_col = (Result == 1)
        FN_col = (Result == -1)
        TP_col = (Result2 == 2)
        TN_col = (Result2 == 0)

        icI_FN[i] = sum(ic*(FN_col.sum(axis = 0)))
        icI_FP[i] = sum(ic*(FP_col.sum(axis = 0)))
        icI_TP[i] = sum(ic*(TP_col.sum(axis = 0)))

        FP[i] = FP_col.sum()
        FN[i] = FN_col.sum()
        TP[i] = TP_col.sum()
        TN[i] = TN_col.sum()

    pr = np.zeros(N_thresholdList)
    rc = np.zeros(N_thresholdList)

    for i in range(N_thresholdList):
        if icI_TP[i] + icI_FP[i] == 0 :
            pr[i] = 0
        else:
            pr[i] = icI_TP[i] / (icI_TP[i] + icI_FP[i])

    rc = icI_TP / (icI_TP + icI_FN)

    F = 2 * pr * rc / (pr + rc + 10e-12)
    Fmax = max(F[np.where(n_e > 0)])

    accuracy = max((TP+TN)/(TP+TN+FP+FN))
    return Fmax, accuracy

def accuracy(y, y_pred):
    thresholdList = np.arange(0, 1.001, 0.001).tolist()
    N_thresholdList = len(thresholdList)

    TP = np.zeros(N_thresholdList)
    TN = np.zeros(N_thresholdList)
    FP = np.zeros(N_thresholdList)
    FN = np.zeros(N_thresholdList)

    Obs = y.copy()
    for i in range(N_thresholdList):
        Pred = (y_pred >= thresholdList[i])
        Result = Pred - Obs
        Result2 = Pred + Obs
        FP_col = (Result == 1)
        FN_col = (Result == -1)
        TP_col = (Result2 == 2)
        TN_col = (Result2 == 0)

        FP[i] = FP_col.sum()
        FN[i] = FN_col.sum()
        TP[i] = TP_col.sum()
        TN[i] = TN_col.sum()



    accuracy = max((TP+TN)/(TP+TN+FP+FN))
    return accuracy


def fold(idx_train,idx_test,feature_matrix,matrix,par):

    print(methodOption + ' starts:')
    X = np.array(feature_matrix[idx_train, :].copy())
    X_new = np.array(feature_matrix[idx_test, :].copy())
    Y = np.array(matrix[idx_train, :].copy())
    y_gt = np.array(matrix[idx_test, :].copy())
    y_new = runModels(Y=Y,X=X,X_new=X_new,method_option=methodOption,par=par)
    print(methodOption + ' ends:')

    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    # keys = ["AUPRperdrug", "AUROCperdrug", "F1perdrug", "accuracy", "AUPR", "AUROC", "F1", "accuracy","adj F1 score"]
    # keys = metrices.keys()
    # results = {key: 0 for key in keys}
    results = {}

    # -----perdrug -----"

    ALL_aupr = np.zeros(len(idx_test))
    ALL_auroc = np.zeros(len(idx_test))
    ALL_F1score = np.zeros(len(idx_test))
    # ALL_accuracy = np.zeros(len(idx_test))
    # ALL_adj_aupr = np.zeros(len(idx_test))
    for i in range(len(idx_test)):
        if y_gt[i, :].sum() == 0:
            ALL_auroc[i] = np.nan
            ALL_aupr[i] = np.nan
            # ALL_F1score[i] = np.nan
        elif y_new[i, :].sum() == 0:
            ALL_auroc[i] = 0
            ALL_aupr[i] = np.nan
            # ALL_F1score[i] = np.nan
        else:
            fpr, tpr, threshold = metrics.roc_curve(y_gt[i, :], y_new[i, :])
            ALL_auroc[i] = auc(fpr, tpr)
            prec, recall, threshold = precision_recall_curve(y_gt[i, :], y_new[i, :])
            # ALL_F1score[i] = max(2*prec*recall/(prec + recall + 10e-8))
            ALL_aupr[i] = auc(recall, prec)
        # ALL_adj_aupr[i] = (auc(recall, prec) - Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]) # / Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]

        # ALL_accuracy[i] = accuracy(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    results["AUPRperdrug"] = np.nanmean(ALL_aupr)
    results["AUROCperdrug"] = np.nanmean(ALL_auroc)
    # results["F1perdrug"]= np.nanmean(ALL_F1score)
    # results["accuracy"] = np.nanmean(ALL_accuracy)
    results["AUPR+AUROCperdrug"] = results["AUPRperdrug"] + results["AUROCperdrug"]

   # ----- overall -----

    prec, recall, prthreshold = precision_recall_curve(y_gt.ravel(), y_new.ravel())
    results["AUPR"] = auc(recall, prec)

    fpr, tpr, rocthreshold = metrics.roc_curve(y_gt.ravel(), y_new.ravel())
    results["AUROC"] = auc(fpr, tpr)

    results["AUPR+AUROC"] = results["AUPR"] + results["AUROC"]


    # results["F1"] = max(2*prec*recall/(prec + recall + 10e-8))

    # likelihood_obs = y_gt.copy()
    # ic = - np.log2((likelihood_obs.sum(axis=0) + 1)/(likelihood_obs.shape[0] + 2))

    # results["adj F1 score"], results["accuracy all"] = adjFmax(y=Ground_Truth[idx_test, :], y_pred=score[idx_test, :], ic=ic)
    # results["adjF1"], _ = adjFmax(y=y_gt, y_pred=y_new, ic=ic)

    print("-----------")
    keys = results.keys()
    for key in keys:
        print(f"{key}: {results[key]}")
    print("-----------")
    
    # ap = average_precision_score(Ground_Truth[target_idx, :].ravel(), score[target_idx, :].ravel())
    # print("Average precision:", ap)
    # print(Ground_Truth[idx_test, :], score[idx_test, :])
    # Plot the heat map
    # sns.heatmap(y_gt, cmap='viridis')
    
    # # Add labels and title
    # plt.xlabel('ADRs')
    # plt.ylabel('drugs')
    
    # # Display the plot
    # row_min = y_new.min(axis=1, keepdims=True)
    # row_max = y_new.max(axis=1, keepdims=True)
    
    # # Normalize each row
    # normalized_matrix = (y_new - row_min) / (row_max - row_min)
    # prec, recall, prthreshold = precision_recall_curve(y_gt.ravel(), normalized_matrix.ravel())
    # print(auc(recall, prec))
    # plt.show()
    # sns.heatmap(normalized_matrix, cmap='viridis')
    
    # # Add labels and title
    # plt.xlabel('ADRs')
    # plt.ylabel('drugs')
    
    # # Display the plot
    # plt.show()


    return results

def innerfold(idx_train,idx_test,feature_matrix,matrix,par):

    X = np.array(feature_matrix[idx_train, :].copy())
    X_new = np.array(feature_matrix[idx_test, :].copy())
    Y = np.array(matrix[idx_train, :].copy())
    y_gt = np.array(matrix[idx_test, :].copy())
    y_new = runModels(Y=Y,X=X,X_new=X_new,method_option=methodOption,par=par)


    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if tuningMetrice == "AUROCperdrug": 
        ALL_auroc = np.zeros(len(idx_test))
        for i in range(len(idx_test)):
            if y_gt[i, :].sum() == 0:
                ALL_auroc[i] = np.nan
            elif y_new[i, :].sum() == 0:
                ALL_auroc[i] = 0
            else:
                fpr, tpr, threshold = metrics.roc_curve(y_gt[i, :], y_new[i, :])
                ALL_auroc[i] = auc(fpr, tpr) 
        result = np.nanmean(ALL_auroc)
    elif tuningMetrice == "AUPRperdrug": 
        ALL_aupr = np.zeros(len(idx_test))
        for i in range(len(idx_test)):
            if y_gt[i, :].sum() == 0:
                ALL_aupr[i] = np.nan
            elif y_new[i, :].sum() == 0:
                ALL_aupr[i] = np.nan
            else:
                prec, recall, threshold = precision_recall_curve(y_gt[i, :], y_new[i, :])
                ALL_aupr[i] = auc(recall, prec)
        result = np.nanmean(ALL_aupr)
    elif tuningMetrice == "AUROC": 
        fpr, tpr, rocthreshold = metrics.roc_curve(y_gt.ravel(), y_new.ravel())
        result = auc(fpr, tpr)
    elif tuningMetrice == "AUPR": 
        # if :
        #     aupr = 0
        # if (Ground_Truth[idx_test, :].ravel().sum() + y_new.ravel().sum() == 0):
        #     result = 0
        if y_new.ravel().sum() == 0:
            result = 0
        else: 
            # prec, recall, threshold = precision_recall_curve(Ground_Truth[idx_test, :].ravel(), score[idx_test, :].ravel())
            prec, recall, threshold = precision_recall_curve(y_gt.ravel(), y_new.ravel())
            result = auc(recall, prec)
    # elif tuningMetrice == "adj AUPRperdrug": 
    #     ALL_aupr = np.zeros(len(idx_test))
    #     ALL_auroc = np.zeros(len(idx_test))
    #     for i in range(len(idx_test)):
    #         if Ground_Truth[idx_test[i], :].sum() + score[idx_test[i], :].sum() == 0:
    #             ALL_aupr[i] = np.nan
    #             ALL_auroc[i] = np.nan
    #             fpr, tpr, threshold = metrics.roc_curve(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             # a = roc_auc_score(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             # print("dc auroc", fpr,tpr)
    #         else:
    #         # print("----")
    #         # print(Ground_Truth[:, idx_test[i]].sum())
    #         # print(score[:, idx_test[i]].sum())
    #             prec, recall, threshold = precision_recall_curve(Ground_Truth[:, idx_test[i]], score[:, idx_test[i]])
    #             ALL_aupr[i] = auc(recall, prec)
    #             # ALL_aupr[i] = (auc(recall, prec) - Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]) # / Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]
    #             fpr, tpr, threshold = metrics.roc_curve(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             ALL_auroc[i] = roc_auc_score(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             # if (ALL_aupr[i] == 1):
    #             #     print("auroc", auc(fpr, tpr), a, "aupr" , ALL_aupr[i], "GT", Ground_Truth[:, idx_test[i]], "pred", score[:, idx_test[i]], fpr, tpr, prec, recall)

    #     # if (par[0] == 0.01) & ((par[1] == 0.01)|(par[1] == 0.1)):
    #     #     print("auroc", np.nanmean(ALL_auroc), "aupr" , np.nanmean(ALL_aupr), par, "\n")
    #     #     pd.DataFrame(Ground_Truth[idx_test, :]).to_csv(f'GT{par[1]}.tsv', sep='\t')
    #     #     pd.DataFrame(score[idx_test, :]).to_csv(f'pred{par[1]}.tsv', sep='\t')


        # result = np.nanmean(ALL_aupr)
    else:
    
        raise ValueError("Please select a metrice for tuning. Choose AUPRperdrug, AUROCperdrug, AUPR or AUROC.")
    return result


def fold3(idx_train,idx_test,feature_matrix,matrix,par):

    print(methodOption + ' starts:')
    X = {}
    X_new = {}
    for i in feature_matrix.keys():
        X = np.array(feature_matrix[i][idx_train, :].copy())
        X_new = np.array(feature_matrix[i][idx_test, :].copy())
    Y = np.array(matrix[idx_train, :].copy())
    y_gt = np.array(matrix[idx_test, :].copy())
    y_new = runModels(Y=Y,X=X,X_new=X_new,method_option=methodOption,par=par)
    print(methodOption + ' ends:')

    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    # keys = ["AUPRperdrug", "AUROCperdrug", "F1perdrug", "accuracy", "AUPR", "AUROC", "F1", "accuracy","adj F1 score"]
    # keys = metrices.keys()
    # results = {key: 0 for key in keys}
    results = {}

    # -----perdrug -----"

    ALL_aupr = np.zeros(len(idx_test))
    ALL_auroc = np.zeros(len(idx_test))
    ALL_F1score = np.zeros(len(idx_test))
    # ALL_accuracy = np.zeros(len(idx_test))
    # ALL_adj_aupr = np.zeros(len(idx_test))
    for i in range(len(idx_test)):
        if y_gt[i, :].sum() == 0:
            ALL_auroc[i] = np.nan
            ALL_aupr[i] = np.nan
            # ALL_F1score[i] = np.nan
        elif y_new[i, :].sum() == 0:
            ALL_auroc[i] = 0
            ALL_aupr[i] = np.nan
            # ALL_F1score[i] = np.nan
        else:
            fpr, tpr, threshold = metrics.roc_curve(y_gt[i, :], y_new[i, :])
            ALL_auroc[i] = auc(fpr, tpr)
            prec, recall, threshold = precision_recall_curve(y_gt[i, :], y_new[i, :])
            # ALL_F1score[i] = max(2*prec*recall/(prec + recall + 10e-8))
            ALL_aupr[i] = auc(recall, prec)
        # ALL_adj_aupr[i] = (auc(recall, prec) - Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]) # / Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]

        # ALL_accuracy[i] = accuracy(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    results["AUPRperdrug"] = np.nanmean(ALL_aupr)
    results["AUROCperdrug"] = np.nanmean(ALL_auroc)
    # results["F1perdrug"]= np.nanmean(ALL_F1score)
    # results["accuracy"] = np.nanmean(ALL_accuracy)
    results["AUPR+AUROCperdrug"] = results["AUPRperdrug"] + results["AUROCperdrug"]

   # ----- overall -----

    prec, recall, prthreshold = precision_recall_curve(y_gt.ravel(), y_new.ravel())
    results["AUPR"] = auc(recall, prec)

    fpr, tpr, rocthreshold = metrics.roc_curve(y_gt.ravel(), y_new.ravel())
    results["AUROC"] = auc(fpr, tpr)

    results["AUPR+AUROC"] = results["AUPR"] + results["AUROC"]


    # results["F1"] = max(2*prec*recall/(prec + recall + 10e-8))

    # likelihood_obs = y_gt.copy()
    # ic = - np.log2((likelihood_obs.sum(axis=0) + 1)/(likelihood_obs.shape[0] + 2))

    # results["adj F1 score"], results["accuracy all"] = adjFmax(y=Ground_Truth[idx_test, :], y_pred=score[idx_test, :], ic=ic)
    # results["adjF1"], _ = adjFmax(y=y_gt, y_pred=y_new, ic=ic)

    print("-----------")
    keys = results.keys()
    for key in keys:
        print(f"{key}: {results[key]}")
    print("-----------")
    
    # ap = average_precision_score(Ground_Truth[target_idx, :].ravel(), score[target_idx, :].ravel())
    # print("Average precision:", ap)
    # print(Ground_Truth[idx_test, :], score[idx_test, :])
    # Plot the heat map
    # sns.heatmap(y_gt, cmap='viridis')
    
    # # Add labels and title
    # plt.xlabel('ADRs')
    # plt.ylabel('drugs')
    
    # # Display the plot
    # row_min = y_new.min(axis=1, keepdims=True)
    # row_max = y_new.max(axis=1, keepdims=True)
    
    # # Normalize each row
    # normalized_matrix = (y_new - row_min) / (row_max - row_min)
    # prec, recall, prthreshold = precision_recall_curve(y_gt.ravel(), normalized_matrix.ravel())
    # print(auc(recall, prec))
    # plt.show()
    # sns.heatmap(normalized_matrix, cmap='viridis')
    
    # # Add labels and title
    # plt.xlabel('ADRs')
    # plt.ylabel('drugs')
    
    # # Display the plot
    # plt.show()


    return results

def innerfold3(idx_train,idx_test,feature_matrix,matrix,par):

    X = {}
    X_new = {}
    for i in feature_matrix.keys():
        X = np.array(feature_matrix[i][idx_train, :].copy())
        X_new = np.array(feature_matrix[i][idx_test, :].copy())
    Y = np.array(matrix[idx_train, :].copy())
    y_gt = np.array(matrix[idx_test, :].copy())
    y_new = runModels(Y=Y,X=X,X_new=X_new,method_option=methodOption,par=par)


    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if tuningMetrice == "AUROCperdrug": 
        ALL_auroc = np.zeros(len(idx_test))
        for i in range(len(idx_test)):
            if y_gt[i, :].sum() == 0:
                ALL_auroc[i] = np.nan
            elif y_new[i, :].sum() == 0:
                ALL_auroc[i] = 0
            else:
                fpr, tpr, threshold = metrics.roc_curve(y_gt[i, :], y_new[i, :])
                ALL_auroc[i] = auc(fpr, tpr) 
        result = np.nanmean(ALL_auroc)
    elif tuningMetrice == "AUPRperdrug": 
        ALL_aupr = np.zeros(len(idx_test))
        for i in range(len(idx_test)):
            if y_gt[i, :].sum() == 0:
                ALL_aupr[i] = np.nan
            elif y_new[i, :].sum() == 0:
                ALL_aupr[i] = np.nan
            else:
                prec, recall, threshold = precision_recall_curve(y_gt[i, :], y_new[i, :])
                ALL_aupr[i] = auc(recall, prec)
        result = np.nanmean(ALL_aupr)
    elif tuningMetrice == "AUROC": 
        fpr, tpr, rocthreshold = metrics.roc_curve(y_gt.ravel(), y_new.ravel())
        result = auc(fpr, tpr)
    elif tuningMetrice == "AUPR": 
        # if :
        #     aupr = 0
        # if (Ground_Truth[idx_test, :].ravel().sum() + y_new.ravel().sum() == 0):
        #     result = 0
        if y_new.ravel().sum() == 0:
            result = 0
        else: 
            # prec, recall, threshold = precision_recall_curve(Ground_Truth[idx_test, :].ravel(), score[idx_test, :].ravel())
            prec, recall, threshold = precision_recall_curve(y_gt.ravel(), y_new.ravel())
            result = auc(recall, prec)
    # elif tuningMetrice == "adj AUPRperdrug": 
    #     ALL_aupr = np.zeros(len(idx_test))
    #     ALL_auroc = np.zeros(len(idx_test))
    #     for i in range(len(idx_test)):
    #         if Ground_Truth[idx_test[i], :].sum() + score[idx_test[i], :].sum() == 0:
    #             ALL_aupr[i] = np.nan
    #             ALL_auroc[i] = np.nan
    #             fpr, tpr, threshold = metrics.roc_curve(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             # a = roc_auc_score(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             # print("dc auroc", fpr,tpr)
    #         else:
    #         # print("----")
    #         # print(Ground_Truth[:, idx_test[i]].sum())
    #         # print(score[:, idx_test[i]].sum())
    #             prec, recall, threshold = precision_recall_curve(Ground_Truth[:, idx_test[i]], score[:, idx_test[i]])
    #             ALL_aupr[i] = auc(recall, prec)
    #             # ALL_aupr[i] = (auc(recall, prec) - Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]) # / Random_AUPR_mean[Ground_Truth[:, idx_test[i]].sum().astype(int)]
    #             fpr, tpr, threshold = metrics.roc_curve(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             ALL_auroc[i] = roc_auc_score(Ground_Truth[idx_test[i], :], score[idx_test[i], :])
    #             # if (ALL_aupr[i] == 1):
    #             #     print("auroc", auc(fpr, tpr), a, "aupr" , ALL_aupr[i], "GT", Ground_Truth[:, idx_test[i]], "pred", score[:, idx_test[i]], fpr, tpr, prec, recall)

    #     # if (par[0] == 0.01) & ((par[1] == 0.01)|(par[1] == 0.1)):
    #     #     print("auroc", np.nanmean(ALL_auroc), "aupr" , np.nanmean(ALL_aupr), par, "\n")
    #     #     pd.DataFrame(Ground_Truth[idx_test, :]).to_csv(f'GT{par[1]}.tsv', sep='\t')
    #     #     pd.DataFrame(score[idx_test, :]).to_csv(f'pred{par[1]}.tsv', sep='\t')


        # result = np.nanmean(ALL_aupr)
    else:
    
        raise ValueError("Please select a metrice for tuning. Choose AUPRperdrug, AUROCperdrug, AUPR or AUROC.")
    return result



def setvar_tune(size, f):
    # set metric hyper pars and the size is the total number of hyperpar combinations
    global ALL_tuning_metrices
    ALL_tuning_metrices = np.zeros((size, f))

def asgvar_tune(idx, f, results):
    # assign var for nested cv from results
    ALL_tuning_metrices[idx, f] = results

def tuning_results(tuneVar):
    # find the best hyperpar combination and its metric value
    mean = ALL_tuning_metrices.mean(axis=1)
    idx = np.argmax(mean)
    Var = tuneVar[idx]
    Value = mean[idx]
    print(f"best hyperpar: {Var}")
    print(f"{tuningMetrice}: {Value}")
    print(f"{tuningMetrice} for each fold: {ALL_tuning_metrices[idx, :]}")

    return Var, Value


# def setvar_besttune(innerfolds):
#     # set var for the cv or nested cv to assign the value for each fold or innerfold
#     global besttunevalue
#     global besttunevar
#     besttunevalue = np.zeros(innerfolds) # best metric value
#     besttunevar = np.zeros(innerfolds) # the value of best var
#     besttunevar = besttunevar.tolist()

# def asg_besttune(f, value, var):
#     # assign var for the cv or nested cv to assign the value for each fold or innerfold
#     besttunevalue[f] = value
#     besttunevar[f] = var

# def besttune():
#     # return be best hyperpar
#     idx = np.argmax(besttunevalue)
#     value = besttunevalue[idx]
#     var = besttunevar[idx]
#     return value, var

def setvar_cv(FOLDS):
# set var for cv 
    global metrices
    # Define the keys for your dictionary
    # metrice_names = ["AUPR", "AUROC", "F1 score", "accuracy", "AUPR all", "AUROC all", "F1 score all", "accuracy all","adj F1 score"]
    # Initialize the dictionary with zero vectors for each key
    # metrices = {metrice_name: np.zeros(FOLDS) for metrice_name in metrice_names}
    metrices = defaultdict(list)

def asgvar_cv(f, results):
    # assign var for cv from results
    # f: size of hyper pars
    keys = results.keys()
    for key in keys:
        metrices[key].append(results[key])


def cv_results():
    keys = metrices.keys()
    for key in keys:
        print(f"Mean {key}: {(np.array(metrices[key])).mean()}, std: {(np.array(metrices[key])).std()}")
    print("-----------")
    return dict(metrices)

# import itertools

# def generate_mean_rd_auprs(y_true):
#     mean_rd_auprs = []
#     m, n = y_true.shape
#     length = n
#     num_ones = y_true.sum(axis=1).astype(int)
#     print(m)
#     for i in range(m):
#         print(i)
#         mean_rd_aupr_i = []
#         # for num_one in range(num_ones[i] + 1):
#         for length_i in range(length + 1, length + 2):
#             # indices_combinations = itertools.combinations(range(length), num_one)
#             indices_combinations = itertools.combinations(range(length), length_i)
#             aupr = []
#             for indices in indices_combinations:
#                 vector = np.zeros(length)
#                 for index in indices:
#                     vector[index] = 1
#                 # print(y_true[i, :])
#                 # print(vector.astype(int))
#                 prec, recall, threshold = precision_recall_curve(y_true[i, :], vector)
                
#                 aupr.append(auc(recall, prec))
#                 # fpr, tpr, rocthreshold = metrics.roc_curve(y_true[i, :], vector)
#                 # auroc = auc(fpr, tpr)
#                 # print(auc(recall, prec))
#                 # aupr = auc(recall, prec)
#             mean_rd_aupr = np.nanmean(aupr)
#             print(mean_rd_aupr)
#             mean_rd_aupr_i.append(mean_rd_aupr)
#         mean_rd_auprs.append(mean_rd_aupr_i)
#     return mean_rd_auprs

# def generate_mean_rd_auprs(y_true):
#     mean_rd_auprs = []
#     m, n = y_true.shape
#     num_ones = y_true.sum(axis=1).astype(int)
#     print(len(num_ones))
#     length = n
#     for i in range(m):
#         print(i)
#         # for num_one in range(num_ones[i] + 1):
#         indices_combinations = itertools.combinations(range(length), num_ones[i])
#         aupr = []
#         for indices in indices_combinations:
#             vector = np.zeros(length)
#             for index in indices:
#                 vector[index] = 1
#             prec, recall, threshold = precision_recall_curve(y_true[i, :], vector)
#             aupr.append(auc(recall, prec))
#             # print(aupr)
#         mean_rd_aupr = np.nanmean(aupr)
#         mean_rd_auprs.append(mean_rd_aupr)
#     return mean_rd_auprs

# def generate_mean_rd_auprs(n, num_ones):
#     warnings.filterwarnings("ignore", category=UserWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     np.random.seed(1949)
#     mean_rd_auprs = []
#     num_rand = 100
#     length = n
#     pred_random = np.random.random((num_rand, n))
#     mean_rd_auprs = []
#     # print(num_ones)
#     for num_one in range(0, num_ones + 1):
#         # print(num_one)
#         vector = [1] * num_one + [0] * (length - num_one)
#         # print(vector)
#         aupr = []
#         for i in range(num_rand):
#             prec, recall, threshold = precision_recall_curve(vector, pred_random[i])
#             aupr.append(auc(recall, prec))
#         mean_rd_auprs.append(np.nanmean(aupr))

#     return mean_rd_auprs


def tuning_loop(innermatrix, idx_train_inner, idx_test_inner, feature_matrix_inner,  hyperparList, i, f):
    
    # print('test set size:', len(idx_test_inner))
    results = innerfold(idx_train_inner,idx_test_inner,feature_matrix=feature_matrix_inner,matrix=innermatrix,par=hyperparList[i])
    asgvar_tune(i, f, results=results)
    # print("------ lmd1: ", l1, "lmd1: ", l2, "sigma: ", s, "------")

def tuning_loop3(innermatrix, idx_train_inner, idx_test_inner, feature_matrix_inner,  hyperparList, i, f):
    
    # print('test set size:', len(idx_test_inner))
    results = innerfold3(idx_train_inner,idx_test_inner,feature_matrix=feature_matrix_inner,matrix=innermatrix,par=hyperparList[i])
    asgvar_tune(i, f, results=results)
    # print("------ lmd1: ", l1, "lmd1: ", l2, "sigma: ", s, "------")

def normalization(K):
    d1 = K.sum(axis=0) + 10e-8
    d2 = K.sum(axis=1) + 10e-8
    K_normalized = (K.T / np.sqrt(d2)).T / np.sqrt(d1)
    return K_normalized

def evaluation(Y, X, method_option, tuning_metrice, hyperparList=[0], hyperparfixed=False, Validation=False,  n_jobs=10):
    random.seed(1949)
    np.random.seed(1949)
    option(method_option, tuning_metrice, Validation)

    matrix_all = np.array(Y.copy()).astype(float)
    m_all,n_all = matrix_all.shape # number of drug # number of side effect 
    drug = Y.index
    
    
    ### Setting validation set / training set / testing set ###
    validate_sz = int(0.25 * m_all)
    IDX_all = list(range(m_all))
    random.shuffle(IDX_all)
    IDX_validate = sorted(IDX_all[0:validate_sz])
    # print("first few validation set idx:")
    # print(IDX_validate[0:10])
    IDX_validate_diff = np.setdiff1d(IDX_all, IDX_validate)
    # matrix = matrix_all[IDX_validate_diff, :].copy().astype(float)

    df = X
    featureMat_all = FeaturePreprocess(df, drug=drug).astype(float)
    # featureMat = featureMat_all[IDX_validate_diff, :].copy().astype(float)
    
    
    # non_zero_idx_union = np.hstack(np.where(~((featureMat.sum(1) == 0) & (featureMat.sum(1) == 0))))
    # non_zero_idx_missing = np.hstack(np.where(~(~(featureMat.sum(1) == 0) & ~(featureMat.sum(1) == 0))))
    # non_zero_idx_intersect = np.hstack(np.where(~(featureMat.sum(1) == 0) & ~(matrix.sum(1) == 0)))

    # intersect
    non_zero_idx_intersect_all = np.hstack(np.where(~(featureMat_all.sum(1) == 0) & ~(matrix_all.sum(1) == 0)))

    # matrix_all = matrix_all[non_zero_idx_intersect_all, :].copy()
    # featureMat_all = featureMat_all[non_zero_idx_intersect_all, :].copy()

    # matrix = matrix[non_zero_idx_intersect, :].copy()
    # featureMat = featureMat[non_zero_idx_intersect, :].copy()

    IDX_validate = np.intersect1d(non_zero_idx_intersect_all, IDX_validate)
    IDX_validate_diff = np.intersect1d(non_zero_idx_intersect_all, IDX_validate_diff)


    distance_X_all = cdist(featureMat_all, featureMat_all)**2
    distance_Y_all = cdist(matrix_all, matrix_all)**2

    # xsc = normalization(distance_X_all)
    # # xsc[xsc < 0.001] = 0
    # ysc = normalization(distance_Y_all)
    # # ysc[ysc < 0.001] = 0

    # plt.scatter(xsc.ravel(), ysc.ravel(), alpha=0.01)
    # plt.title('Sample Scatter Plot')
    # plt.xlabel('K_X')
    # plt.ylabel('K_Y')
    # plt.show()


    # drug_nodes_intersect_all = np.array(drug)[non_zero_idx_intersect_all]
    # drug_nodes_intersect_validate_diff = np.array(drug)[IDX_validate_diff]
    # drug_nodes_intersect_validate = np.array(drug)[IDX_validate]

    # IDX_validate = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate])
    # IDX_validate_diff = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate_diff])
    

    # m,n = matrix.shape  # number of drug # number of side effect


    FOLDS = 5
    innerFOLDS = 4
    ####for test sets####
    setvar_cv(FOLDS)

    sz = len(IDX_validate_diff)
    IDX = list(range(sz))
    fsz = int(sz/FOLDS)
    random.shuffle(IDX)
    IDX = np.array(IDX)
    offset = 0

    innersz = sz - fsz
    innerIDX = list(range(innersz))
    random.shuffle(innerIDX)
    innerIDX = np.array(innerIDX)
    innerfsz = int(innersz / innerFOLDS)
    inneroffset = 0
    # setvar_cv(FOLDS=FOLDS)
    # column_sums_all = np.sum(matrix_all, axis=0)
    # matrix_all = matrix_all[:, (column_sums_all >= 15)&(column_sums_all < 10000)].copy()

    # column_sums = np.sum(matrix, axis=0)
    # matrix = matrix[:, (column_sums >= 15)&(column_sums < 10000)].copy()

    # global Random_AUPR_mean
    # Random_AUPR_mean = generate_mean_rd_auprs(n=n, num_ones=max(matrix_all.sum(axis=1)).astype(int))
    if Validation == "nested_cv":
        bestHyperParsOut=[]
        print("---------- nested cv start ----------")
        for f in range(FOLDS):
            offset = 0 + fsz*f 
            # idx_test = IDX[offset:offset + fsz]
            # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
            # idx_train = IDX_validate_diff[np.sort(IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))])]
            idx_test = IDX_validate_diff[IDX[offset:offset + fsz]]    
            idx_train = np.setdiff1d(IDX_validate_diff, idx_test)
            print("Fold:",f)

            setvar_tune(len(hyperparList), f = innerFOLDS)
            print("number of hyperpars combination: ", len(hyperparList))
            print("first few training idx: ", np.sort(idx_train[0:10]))
            print("first few testing idx: ", np.sort(idx_test[0:10]))
            if hyperparfixed == False:
                for innerf in range(innerFOLDS):
                    # inneroffset = 0 + innerf*innerfsz
                    # idx_test_inner = innerIDX[inneroffset:inneroffset + innerfsz]
                    # idx_train_inner = innerIDX[np.array(np.setdiff1d(np.arange(len(idx_train)), np.arange(inneroffset,  inneroffset + innerfsz)))]
                    inneroffset = 0 + innerf*innerfsz
                    idx_test_inner = idx_train[innerIDX[inneroffset:inneroffset + innerfsz]]
                    idx_train_inner = np.setdiff1d(idx_train, idx_test_inner)
                    

                    # print("first few training idx: ", np.sort(idx_train_inner[0:10]))
                    # print("first few testing idx: ", np.sort(idx_test_inner[0:10]))



                    setDist(dist_X=distance_X_all[np.ix_(idx_train_inner, idx_train_inner)], dist_Y=distance_Y_all[np.ix_(idx_train_inner, idx_train_inner)], dist_X_new=distance_X_all[np.ix_(idx_test_inner, idx_train_inner)])
                    # setDist()
    
                    print("Inner Fold:", innerf)
                    with parallel_backend('threading'):
                        Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix_all, idx_train_inner = idx_train_inner, idx_test_inner = idx_test_inner, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=innerf) for i in range(len(hyperparList)))
                    setDist()
                bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)
                bestHyperParsOut.append(bestHyperPars)
            else:
                setDist()
                bestHyperPars = hyperparfixed[f]
                bestHyperParsOut.append(bestHyperPars)
            
    
            # asg_besttune(innerf, value=evalValue, var=hyperpars)
                # raise ValueError("--.")
                    
            # _, bestHyperPars = besttune()
    
            print("--- tuning end ---")
            print('target size:', len(idx_test))
            print("------ best hyper pars: ", bestHyperPars, "------")
            results = fold(idx_train,idx_test,featureMat_all,matrix_all,par=bestHyperPars)
            asgvar_cv(f=f, results=results)

        out = cv_results()
        return out, bestHyperParsOut

    elif Validation == "cv":
        print("---------- cv start ----------")
        # setvar_besttune(FOLDS)

        setvar_tune(len(hyperparList), f = FOLDS)
        if hyperparfixed == False:
            
            for f in range(FOLDS):
                offset = 0 + fsz*f 
                # idx_test = IDX[offset:offset + fsz]
                # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]


                idx_test = IDX_validate_diff[IDX[offset:offset + fsz]]    
                idx_train = np.setdiff1d(IDX_validate_diff, idx_test)
                setDist(dist_X=distance_X_all[np.ix_(idx_train, idx_train)], dist_Y=distance_Y_all[np.ix_(idx_train, idx_train)], dist_X_new=distance_X_all[np.ix_(idx_test, idx_train)])
                # setDist()

                print("Fold:", f)

                with parallel_backend('threading'):
                    Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix_all, idx_train_inner = idx_train, idx_test_inner = idx_test, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=f)for i in range(len(hyperparList)))
                setDist()
            bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)

        else:
            setDist()
            bestHyperPars = hyperparfixed

        # asg_besttune(f, value=evalValue, var=hyperpars)

    
        print("--- tuning end ---")
        # cv_results()
        # _, bestHyperPars = besttune()
        # validation
        idx_test = IDX_validate
        idx_train = IDX_validate_diff
        print('target size:', len(idx_test))
        print("------ best hyper pars: ", bestHyperPars, "------")
        results = fold(idx_train,idx_test,featureMat_all,matrix_all,par=bestHyperPars)
        return results, bestHyperPars


def evaluation2(Y, X, X2, method_option, tuning_metrice, hyperparList=[0], hyperparfixed=False, Validation=False,  n_jobs=10):
    random.seed(1949)
    np.random.seed(1949)
    option(method_option, tuning_metrice, Validation)

    matrix_all = np.array(Y.copy()).astype(float)
    m_all,n_all = matrix_all.shape # number of drug # number of side effect 
    drug = Y.index
    
    
    ### Setting validation set / training set / testing set ###
    validate_sz = int(0.25 * m_all)
    IDX_all = list(range(m_all))
    random.shuffle(IDX_all)
    IDX_validate = sorted(IDX_all[0:validate_sz])
    # print("first few validation set idx:")
    # print(IDX_validate[0:10])
    IDX_validate_diff = np.setdiff1d(IDX_all, IDX_validate)
    # matrix = matrix_all[IDX_validate_diff, :].copy().astype(float)

    df = X
    featureMat_all_0 = FeaturePreprocess(df, drug=drug).astype(float)
    # featureMat = featureMat_all[IDX_validate_diff, :].copy().astype(float)
    
    
    # non_zero_idx_union = np.hstack(np.where(~((featureMat.sum(1) == 0) & (featureMat.sum(1) == 0))))
    # non_zero_idx_missing = np.hstack(np.where(~(~(featureMat.sum(1) == 0) & ~(featureMat.sum(1) == 0))))
    # non_zero_idx_intersect = np.hstack(np.where(~(featureMat.sum(1) == 0) & ~(matrix.sum(1) == 0)))

    # intersect
    non_zero_idx_intersect_all_0 = np.hstack(np.where(~(featureMat_all_0.sum(1) == 0) & ~(matrix_all.sum(1) == 0)))

    # matrix_all = matrix_all[non_zero_idx_intersect_all, :].copy()
    # featureMat_all = featureMat_all[non_zero_idx_intersect_all, :].copy()

    # matrix = matrix[non_zero_idx_intersect, :].copy()
    # featureMat = featureMat[non_zero_idx_intersect, :].copy()
    df = X2
    featureMat_all = FeaturePreprocess(df, drug=drug).astype(float)
    non_zero_idx_intersect_all = np.hstack(np.where(~(featureMat_all.sum(1) == 0) & ~(matrix_all.sum(1) == 0)))

    IDX_validate_0 = np.intersect1d(non_zero_idx_intersect_all_0, IDX_validate)
    IDX_validate_diff = np.intersect1d(non_zero_idx_intersect_all, IDX_validate_diff)
    IDX_validate_diff_0 = np.intersect1d(non_zero_idx_intersect_all_0, IDX_validate_diff)


    distance_X_all = cdist(featureMat_all, featureMat_all)**2
    distance_Y_all = cdist(matrix_all, matrix_all)**2

    # xsc = normalization(distance_X_all)
    # # xsc[xsc < 0.001] = 0
    # ysc = normalization(distance_Y_all)
    # # ysc[ysc < 0.001] = 0

    # plt.scatter(xsc.ravel(), ysc.ravel(), alpha=0.01)
    # plt.title('Sample Scatter Plot')
    # plt.xlabel('K_X')
    # plt.ylabel('K_Y')
    # plt.show()


    # drug_nodes_intersect_all = np.array(drug)[non_zero_idx_intersect_all]
    # drug_nodes_intersect_validate_diff = np.array(drug)[IDX_validate_diff]
    # drug_nodes_intersect_validate = np.array(drug)[IDX_validate]

    # IDX_validate = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate])
    # IDX_validate_diff = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate_diff])
    

    # m,n = matrix.shape  # number of drug # number of side effect


    FOLDS = 5
    innerFOLDS = 4
    ####for test sets####
    setvar_cv(FOLDS)

    sz_0 = len(IDX_validate_diff_0)
    IDX_0 = list(range(sz_0))
    fsz_0 = int(sz_0/FOLDS)
    random.shuffle(IDX_0)
    IDX_0 = np.array(IDX_0)
    offset_0 = 0

    innersz_0 = sz_0 - fsz_0
    innerIDX_0 = list(range(innersz_0))
    random.shuffle(innerIDX_0)
    innerIDX_0 = np.array(innerIDX_0)
    innerfsz_0 = int(innersz_0 / innerFOLDS)
    inneroffset_0 = 0

    sz = len(IDX_validate_diff)
    IDX = list(range(sz))
    fsz = int(sz/FOLDS)
    random.shuffle(IDX)
    IDX = np.array(IDX)
    offset = 0

    innersz = sz - fsz
    innerIDX = list(range(innersz))
    random.shuffle(innerIDX)
    innerIDX = np.array(innerIDX)
    innerfsz = int(innersz / innerFOLDS)
    inneroffset = 0
    # setvar_cv(FOLDS=FOLDS)
    # column_sums_all = np.sum(matrix_all, axis=0)
    # matrix_all = matrix_all[:, (column_sums_all >= 15)&(column_sums_all < 10000)].copy()

    # column_sums = np.sum(matrix, axis=0)
    # matrix = matrix[:, (column_sums >= 15)&(column_sums < 10000)].copy()

    # global Random_AUPR_mean
    # Random_AUPR_mean = generate_mean_rd_auprs(n=n, num_ones=max(matrix_all.sum(axis=1)).astype(int))
    if Validation == "nested_cv":
        bestHyperParsOut=[]
        print("---------- nested cv start ----------")
        for f in range(FOLDS):
            offset_0 = 0 + fsz_0*f 
            # idx_test = IDX[offset:offset + fsz]
            # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
            # idx_test = IDX_validate_diff[IDX[offset:offset + fsz]]    
            # idx_train = IDX_validate_diff[IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]]
            idx_test_0 = IDX_validate_diff_0[IDX_0[offset_0:offset_0 + fsz_0]] 
            idx_train_0 = np.setdiff1d(IDX_validate_diff_0, idx_test_0)  
            idx_train = np.setdiff1d(IDX_validate_diff, idx_test_0)
            print("Fold:",f)

            setvar_tune(len(hyperparList), f = innerFOLDS)
            print("number of hyperpars combination: ", len(hyperparList))
            print("first few training idx: ", np.sort(idx_train[0:10]))
            print("first few testing idx: ", np.sort(idx_test_0[0:10]))
            if hyperparfixed == False:
                for innerf in range(innerFOLDS):
                    # inneroffset = 0 + innerf*innerfsz
                    # idx_test_inner = innerIDX[inneroffset:inneroffset + innerfsz]
                    # idx_train_inner = innerIDX[np.array(np.setdiff1d(np.arange(len(idx_train)), np.arange(inneroffset,  inneroffset + innerfsz)))]
                    inneroffset_0 = 0 + innerf*innerfsz_0
                    idx_test_inner_0 = idx_train_0[innerIDX_0[inneroffset_0:inneroffset_0 + innerfsz_0]]
                    idx_train_inner = np.setdiff1d(idx_train, idx_test_inner_0)
                    # print("first few training idx: ", np.sort(idx_train_inner[0:10]))
                    # print("first few testing idx: ", np.sort(idx_test_inner[0:10]))



                    setDist(dist_X=distance_X_all[np.ix_(idx_train_inner, idx_train_inner)], dist_Y=distance_Y_all[np.ix_(idx_train_inner, idx_train_inner)], dist_X_new=distance_X_all[np.ix_(idx_test_inner_0, idx_train_inner)])
                    # setDist()
    
                    print("Inner Fold:", innerf)
                    with parallel_backend('threading'):
                        Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix_all, idx_train_inner = idx_train_inner, idx_test_inner = idx_test_inner_0, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=innerf) for i in range(len(hyperparList)))
                    setDist()
                bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)
                bestHyperParsOut.append(bestHyperPars)
            else:
                setDist()
                bestHyperPars = hyperparfixed[f]
                bestHyperParsOut.append(bestHyperPars)
            
    
            # asg_besttune(innerf, value=evalValue, var=hyperpars)
                # raise ValueError("--.")
                    
            # _, bestHyperPars = besttune()
    
            print("--- tuning end ---")
            print('target size:', len(idx_test_0))
            print("------ best hyper pars: ", bestHyperPars, "------")
            results = fold(idx_train,idx_test_0,featureMat_all,matrix_all,par=bestHyperPars)
            asgvar_cv(f=f, results=results)

        out = cv_results()
        return out, bestHyperParsOut

    elif Validation == "cv":
        print("---------- cv start ----------")
        # setvar_besttune(FOLDS)

        setvar_tune(len(hyperparList), f = FOLDS)
        if hyperparfixed == False:
            
            for f in range(FOLDS):
                offset_0 = 0 + fsz_0*f 
                # idx_test = IDX[offset:offset + fsz]
                # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]

                idx_test_0 = IDX_validate_diff_0[IDX_0[offset_0:offset_0 + fsz_0]]
                # idx_train_0 = np.setdiff1d(IDX_validate_diff_0, idx_test_0)
                idx_train = np.setdiff1d(IDX_validate_diff, idx_test_0)

                setDist(dist_X=distance_X_all[np.ix_(idx_train, idx_train)], dist_Y=distance_Y_all[np.ix_(idx_train, idx_train)], dist_X_new=distance_X_all[np.ix_(idx_test_0, idx_train)])
                # setDist()

                print("Fold:", f)

                with parallel_backend('threading'):
                    Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix_all, idx_train_inner = idx_train, idx_test_inner = idx_test_0, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=f)for i in range(len(hyperparList)))
                setDist()
            bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)

        else:
            setDist()
            bestHyperPars = hyperparfixed

        # asg_besttune(f, value=evalValue, var=hyperpars)

    
        print("--- tuning end ---")
        # cv_results()
        # _, bestHyperPars = besttune()
        # validation
        idx_test_0 = IDX_validate_0
        idx_train = IDX_validate_diff
        print('target size:', len(idx_test_0))
        print("------ best hyper pars: ", bestHyperPars, "------")
        results = fold(idx_train,idx_test_0,featureMat_all,matrix_all,par=bestHyperPars)
        return results, bestHyperPars
    

def evaluation3(Y, X, X2, method_option, tuning_metrice, hyperparList=[0], hyperparfixed=False, Validation=False,  n_jobs=10):
    random.seed(1949)
    np.random.seed(1949)
    option(method_option, tuning_metrice, Validation)

    matrix_all = np.array(Y.copy()).astype(float)
    m_all,n_all = matrix_all.shape # number of drug # number of side effect 
    drug = Y.index
    
    
    ### Setting validation set / training set / testing set ###
    validate_sz = int(0.25 * m_all)
    IDX_all = list(range(m_all))
    random.shuffle(IDX_all)
    IDX_validate = sorted(IDX_all[0:validate_sz])
    # print("first few validation set idx:")
    # print(IDX_validate[0:10])
    IDX_validate_diff = np.setdiff1d(IDX_all, IDX_validate)
    # matrix = matrix_all[IDX_validate_diff, :].copy().astype(float)
    featureMat_all_0 = {}
    non_zero_idx_intersect_all_0 = np.hstack(np.where(~(matrix_all.sum(1) == 0)))
    for i in X.keys():
        df = X[i]
        featureMat_all_0[i] = FeaturePreprocess(df, drug=drug).astype(float)
        non_zero_idx_intersect_all_0 = np.intersect1d(np.hstack(np.where(~(featureMat_all_0[i].sum(1) == 0))), non_zero_idx_intersect_all_0)
    # featureMat = featureMat_all[IDX_validate_diff, :].copy().astype(float)
    
    
    # non_zero_idx_union = np.hstack(np.where(~((featureMat.sum(1) == 0) & (featureMat.sum(1) == 0))))
    # non_zero_idx_missing = np.hstack(np.where(~(~(featureMat.sum(1) == 0) & ~(featureMat.sum(1) == 0))))
    # non_zero_idx_intersect = np.hstack(np.where(~(featureMat.sum(1) == 0) & ~(matrix.sum(1) == 0)))

    # intersect

    # matrix_all = matrix_all[non_zero_idx_intersect_all, :].copy()
    # featureMat_all = featureMat_all[non_zero_idx_intersect_all, :].copy()

    # matrix = matrix[non_zero_idx_intersect, :].copy()
    # featureMat = featureMat[non_zero_idx_intersect, :].copy()
    # new features
    featureMat_all = {}
    non_zero_idx_intersect_all = np.hstack(np.where(~(matrix_all.sum(1) == 0)))
    for i in X2.keys():
        df = X2[i]
        featureMat_all[i] = FeaturePreprocess(df, drug=drug).astype(float)
        non_zero_idx_intersect_all = np.intersect1d(np.hstack(np.where(~(featureMat_all[i].sum(1) == 0))), non_zero_idx_intersect_all)
        # distance_X_all = cdist(featureMat_all, featureMat_all)**2
        # distance_Y_all = cdist(matrix_all, matrix_all)**2
    

    IDX_validate_0 = np.intersect1d(non_zero_idx_intersect_all_0, IDX_validate)
    IDX_validate_diff = np.intersect1d(non_zero_idx_intersect_all, IDX_validate_diff)
    IDX_validate_diff_0 = np.intersect1d(non_zero_idx_intersect_all_0, IDX_validate_diff)



    # xsc = normalization(distance_X_all)
    # # xsc[xsc < 0.001] = 0
    # ysc = normalization(distance_Y_all)
    # # ysc[ysc < 0.001] = 0

    # plt.scatter(xsc.ravel(), ysc.ravel(), alpha=0.01)
    # plt.title('Sample Scatter Plot')
    # plt.xlabel('K_X')
    # plt.ylabel('K_Y')
    # plt.show()


    # drug_nodes_intersect_all = np.array(drug)[non_zero_idx_intersect_all]
    # drug_nodes_intersect_validate_diff = np.array(drug)[IDX_validate_diff]
    # drug_nodes_intersect_validate = np.array(drug)[IDX_validate]

    # IDX_validate = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate])
    # IDX_validate_diff = np.array([x for x in range(len(drug_nodes_intersect_all)) if drug_nodes_intersect_all[x] in drug_nodes_intersect_validate_diff])
    

    # m,n = matrix.shape  # number of drug # number of side effect


    FOLDS = 5
    innerFOLDS = 4
    ####for test sets####
    setvar_cv(FOLDS)

    sz_0 = len(IDX_validate_diff_0)
    IDX_0 = list(range(sz_0))
    fsz_0 = int(sz_0/FOLDS)
    random.shuffle(IDX_0)
    IDX_0 = np.array(IDX_0)
    offset_0 = 0

    innersz_0 = sz_0 - fsz_0
    innerIDX_0 = list(range(innersz_0))
    random.shuffle(innerIDX_0)
    innerIDX_0 = np.array(innerIDX_0)
    innerfsz_0 = int(innersz_0 / innerFOLDS)
    inneroffset_0 = 0

    sz = len(IDX_validate_diff)
    IDX = list(range(sz))
    fsz = int(sz/FOLDS)
    random.shuffle(IDX)
    IDX = np.array(IDX)
    offset = 0

    innersz = sz - fsz
    innerIDX = list(range(innersz))
    random.shuffle(innerIDX)
    innerIDX = np.array(innerIDX)
    innerfsz = int(innersz / innerFOLDS)
    inneroffset = 0

    if Validation == "nested_cv":
        bestHyperParsOut=[]
        print("---------- nested cv start ----------")
        for f in range(FOLDS):
            offset_0 = 0 + fsz_0*f 
            # idx_test = IDX[offset:offset + fsz]
            # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
            # idx_test = IDX_validate_diff[IDX[offset:offset + fsz]]    
            # idx_train = IDX_validate_diff[IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]]
            idx_test_0 = IDX_validate_diff_0[IDX_0[offset_0:offset_0 + fsz_0]] 
            idx_train_0 = np.setdiff1d(IDX_validate_diff_0, idx_test_0)  
            idx_train = np.setdiff1d(IDX_validate_diff, idx_test_0)
            print("Fold:",f)

            setvar_tune(len(hyperparList), f = innerFOLDS)
            print("number of hyperpars combination: ", len(hyperparList))
            print("first few training idx: ", np.sort(idx_train[0:10]))
            print("first few testing idx: ", np.sort(idx_test_0[0:10]))
            if hyperparfixed == False:
                for innerf in range(innerFOLDS):
                    # inneroffset = 0 + innerf*innerfsz
                    # idx_test_inner = innerIDX[inneroffset:inneroffset + innerfsz]
                    # idx_train_inner = innerIDX[np.array(np.setdiff1d(np.arange(len(idx_train)), np.arange(inneroffset,  inneroffset + innerfsz)))]
                    inneroffset_0 = 0 + innerf*innerfsz_0
                    idx_test_inner_0 = idx_train_0[innerIDX_0[inneroffset_0:inneroffset_0 + innerfsz_0]]
                    idx_train_inner = np.setdiff1d(idx_train, idx_test_inner_0)
                    # print("first few training idx: ", np.sort(idx_train_inner[0:10]))
                    # print("first few testing idx: ", np.sort(idx_test_inner[0:10]))



                    # setDist(dist_X=distance_X_all[np.ix_(idx_train_inner, idx_train_inner)], dist_Y=distance_Y_all[np.ix_(idx_train_inner, idx_train_inner)], dist_X_new=distance_X_all[np.ix_(idx_test_inner_0, idx_train_inner)])
                    setDist()
    
                    print("Inner Fold:", innerf)
                    with parallel_backend('threading'):
                        Parallel(n_jobs=n_jobs)(delayed(tuning_loop3)(innermatrix = matrix_all, idx_train_inner = idx_train_inner, idx_test_inner = idx_test_inner_0, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=innerf) for i in range(len(hyperparList)))
                    setDist()
                bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)
                bestHyperParsOut.append(bestHyperPars)
            else:
                setDist()
                bestHyperPars = hyperparfixed[f]
                bestHyperParsOut.append(bestHyperPars)
            
    
            # asg_besttune(innerf, value=evalValue, var=hyperpars)
                # raise ValueError("--.")
                    
            # _, bestHyperPars = besttune()
    
            print("--- tuning end ---")
            print('target size:', len(idx_test_0))
            print("------ best hyper pars: ", bestHyperPars, "------")
            results = fold3(idx_train,idx_test_0,featureMat_all,matrix_all,par=bestHyperPars)
            asgvar_cv(f=f, results=results)

        out = cv_results()
        return out, bestHyperParsOut

    elif Validation == "cv":
        print("---------- cv start ----------")
        # setvar_besttune(FOLDS)

        setvar_tune(len(hyperparList), f = FOLDS)
        if hyperparfixed == False:
            
            for f in range(FOLDS):
                offset_0 = 0 + fsz_0*f 
                # idx_test = IDX[offset:offset + fsz]
                # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]

                idx_test_0 = IDX_validate_diff_0[IDX_0[offset_0:offset_0 + fsz_0]]
                # idx_train_0 = np.setdiff1d(IDX_validate_diff_0, idx_test_0)
                idx_train = np.setdiff1d(IDX_validate_diff, idx_test_0)

                # setDist(dist_X=distance_X_all[np.ix_(idx_train, idx_train)], dist_Y=distance_Y_all[np.ix_(idx_train, idx_train)], dist_X_new=distance_X_all[np.ix_(idx_test_0, idx_train)])
                setDist()

                print("Fold:", f)

                with parallel_backend('threading'):
                    Parallel(n_jobs=n_jobs)(delayed(tuning_loop3)(innermatrix = matrix_all, idx_train_inner = idx_train, idx_test_inner = idx_test_0, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=f)for i in range(len(hyperparList)))
                setDist()
            bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)

        else:
            setDist()
            bestHyperPars = hyperparfixed

        # asg_besttune(f, value=evalValue, var=hyperpars)

    
        print("--- tuning end ---")
        # cv_results()
        # _, bestHyperPars = besttune()
        # validation
        idx_test_0 = IDX_validate_0
        idx_train = IDX_validate_diff
        print('target size:', len(idx_test_0))
        print("------ best hyper pars: ", bestHyperPars, "------")
        results = fold3(idx_train,idx_test_0,featureMat_all,matrix_all,par=bestHyperPars)
        return results, bestHyperPars
