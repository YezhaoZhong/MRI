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

from ADRprofilePrediction import FeaturePreprocess
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import copy


random.seed(1949)
np.random.seed(1949)


def option(str1, str2, str3):
    global methodOption
    global tuningMetrice
    global validationOption

    methodOption = str1
    tuningMetrice = str2
    validationOption = str3


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





def fold(idx_train,idx_test,feature_matrix,matrix,par):
    X = copy.deepcopy(feature_matrix)
    X_gt = copy.deepcopy(feature_matrix)
    W = copy.deepcopy(feature_matrix)
    shape = {}

    print(methodOption + ' starts:')
    # c = par[1]
    for i in X.keys():
        # X[i] = X[i]*0
        # X[i][idx_train[i], :] = copy.deepcopy(X_gt[i][idx_train[i], :])
        # W[i] = W[i]*0+1*c
        # W[i][idx_train[i], :] = 1
        X[i][idx_test[i], :] = 0
        # W[i][idx_test[i], :] = 0
        # W[i][idx_train[i], :] = 1
        W[i][(W[i] == 0).all(axis=1)] = np.nan
        W[i][idx_test[i], :] = np.nan
        shape[i] = X[i].shape
    
    X_stack = np.concatenate(list(X.values()), axis=1)
    W_stack = np.concatenate(list(W.values()), axis=1)
    # print(sum(X_stack.sum(1) == 0))
    print('concatenated feature shape: ', X_stack.shape)

    X_pred = runModels(Y=None,X=X_stack,X_new=W_stack,method_option=methodOption,par=par)

    print(methodOption + ' ends:')
    X_new = {}
    previous_shape = 0
    results = {}
    gt = []
    pred = []
    for i in X.keys():
        X_new[i] = X_pred[:, previous_shape:previous_shape + shape[i][1]]
        previous_shape = previous_shape + shape[i][1]
        gt = np.append(gt, X_gt[i][idx_test[i], :].ravel())
        pred = np.append(pred, X_new[i][idx_test[i], :].ravel())
    
    results["mse"] = np.mean((gt - pred) ** 2)
    prec, recall, prthreshold = precision_recall_curve(gt, pred)
    results["AUPR"] = auc(recall, prec)

    fpr, tpr, rocthreshold = metrics.roc_curve(gt, pred)
    results["AUROC"] = auc(fpr, tpr)

    # results["AUPR+AUROC"] = results["AUPR"] + results["AUROC"]


    print("-----------")
    keys = results.keys()
    for key in keys:
        print(f"{key}: {results[key]}")
    print("-----------")

    return results, X_new

def innerfold(idx_train,idx_test,feature_matrix,matrix,par):
    # for i in feature_matrix.keys():
    #     print("-")
    #     print(feature_matrix[i][idx_test[i],:].sum(1))

    X = copy.deepcopy(feature_matrix)
    X_gt = copy.deepcopy(feature_matrix)
    W = copy.deepcopy(feature_matrix)
    shape = {}

    # print(methodOption + ' starts:')
    for i in X.keys():
        # c = par[1]
        # print((X[i][idx_test[i], :]).sum())
        # print(X[i][idx_test[i], :].shape)
        # print("+++")
        X[i][idx_test[i], :] = 0
        W[i][(W[i] == 0).all(axis=1)] = np.nan
        W[i][idx_test[i], :] = np.nan
        # X[i] = X[i]*0
        # X[i][idx_train[i], :] = copy.deepcopy(X_gt[i][idx_train[i], :])

        # W[i] = W[i]*0
        # W[i] = W[i]*0+1*c
        # W[i][idx_test[i], :] = 0
        # W[i][idx_train[i], :] = 1
        shape[i] = X[i].shape
        # print("-")
        # print(feature_matrix[i][idx_test[i],:].sum(1))
    
    X_stack = np.concatenate(list(X.values()), axis=1)
    W_stack = np.concatenate(list(W.values()), axis=1)

    X_pred = runModels(Y=None,X=X_stack,X_new=W_stack,method_option=methodOption,par=par)

    # print(methodOption + ' ends:')
    X_new = {}
    previous_shape = 0
    gt = []
    pred = []
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for i in X.keys():
        X_new[i] = X_pred[:, previous_shape:previous_shape + shape[i][1]]
        previous_shape = previous_shape + shape[i][1]
        gt = np.append(gt, X_gt[i][idx_test[i], :].ravel())
        pred = np.append(pred, X_new[i][idx_test[i], :].ravel())
        # print(sum(X_gt[i][idx_test[i], :].ravel()))
        # print(sum(X_new[i][idx_test[i], :].ravel()))

    
    if tuningMetrice == "mse":
        result = -np.mean((gt - pred) ** 2)
    elif tuningMetrice == "AUROC": 
        fpr, tpr, rocthreshold = metrics.roc_curve(gt, pred)
        result = auc(fpr, tpr)
    elif tuningMetrice == "AUPR": 
        if pred.sum() == 0:
            result = 0
        else:
            prec, recall, threshold = precision_recall_curve(gt, pred)
            result = auc(recall, prec)
    else:
    
        raise ValueError("Please select a metrice for tuning. Choose mse.")
    return result, X_new

def completionfold(idx_train,idx_test,feature_matrix,par):
    X = copy.deepcopy(feature_matrix)
    W = copy.deepcopy(feature_matrix)
    shape = {}

    print(methodOption + ' starts:')
    # c = par[1]
    for i in X.keys():
        shape[i] = X[i].shape
        # W[i][idx_test[i], :] = 0
        # W[i][idx_train[i], :] = 1*c
        W[i][(W[i] == 0).all(axis=1)] = np.nan
        # W[i][idx_test[i], :] = np.nan
    X_stack = np.concatenate(list(X.values()), axis=1)
    W_stack = np.concatenate(list(W.values()), axis=1)

    X_pred = runModels(Y=None,X=X_stack,X_new=W_stack,method_option=methodOption,par=par)

    print(methodOption + ' ends:')
    X_new = {}
    previous_shape = 0
    for i in X.keys():
        X_new[i] = X_pred[:, previous_shape:previous_shape + shape[i][1]]
        previous_shape = previous_shape + shape[i][1]

    return X_new


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


def setvar_cv(FOLDS):
    global metrices
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


def tuning_loop(innermatrix, idx_train_inner, idx_test_inner, feature_matrix_inner, hyperparList, i, f):
    
    # print('test set size:', len(idx_test_inner))
    results, _ = innerfold(idx_train_inner,idx_test_inner,feature_matrix=feature_matrix_inner,matrix=innermatrix,par=hyperparList[i])
    asgvar_tune(i, f, results=results)
    # print("------ lmd1: ", l1, "lmd1: ", l2, "sigma: ", s, "------")

def completion(X, Y, method_option, tuning_metrice, hyperparList=[0], hyperparfixed=False, Validation=False,  n_jobs=10):
    random.seed(1949)
    np.random.seed(1949)
    option(method_option, tuning_metrice, Validation)
    matrix_all = np.array(Y.copy()).astype(float)
    m_Y,n_Y = matrix_all.shape # number of drug # number of side effect 
    drug = Y.index
    
    ### Setting validation set / training set / testing set ###
    featureMat_all = {}
    non_zero_idx_intersect_all = {}
    zero_idx_intersect_all = {}
    IDX_validate = {}
    IDX_validate_diff = {}
    innerfsz = {}
    innerIDX = {}
    fsz = {}
    IDX = {}
    inneroffset = {}
    offset = {}

    FOLDS = 5
    innerFOLDS = 4
    ####for test sets####
    setvar_cv(FOLDS)
    IDX_all = list(range(m_Y))
    for i in X.keys():

        df = X[i].loc[np.intersect1d(X[i].index, drug)].copy()
        featureMat_all[i] = FeaturePreprocess(df, drug=drug).astype(float)
        # m_X,n_X = np.array(df).shape # number of drug # number of side effect 
        validate_sz = int(0.25 * m_Y)
        random.shuffle(IDX_all)
        IDX_validate_all = sorted(IDX_all[0:validate_sz])
        IDX_validate_diff_all = np.setdiff1d(IDX_all, IDX_validate_all)
        # intersect
        non_zero_idx_intersect_all[i] = np.hstack(np.where(~(featureMat_all[i].sum(1) == 0)))
        zero_idx_intersect_all[i] = np.hstack(np.where((featureMat_all[i].sum(1) == 0)))

        IDX_validate[i] = np.intersect1d(non_zero_idx_intersect_all[i], IDX_validate_all)
        IDX_validate_diff[i] = np.intersect1d(non_zero_idx_intersect_all[i], IDX_validate_diff_all)

        # print("---")
        # print(len(IDX_validate[i]))
        # print(len(IDX_validate_diff[i]))
    
    # remove the common rows to avoid zero rows in training
    common_numbers = set(IDX_validate[next(iter(IDX_validate))])
    for lst in IDX_validate.values():
        common_numbers.intersection_update(lst)
    if common_numbers:
        for key in IDX_validate:
            IDX_validate[key] = [num for num in IDX_validate[key] if num not in common_numbers]
            IDX_validate_diff[key].extend(common_numbers)
    print("common drugs chosen to be the test set: ", common_numbers)

    for i in X.keys():
        offset[i] = 0
        inneroffset[i] = 0
        sz = len(IDX_validate_diff[i])
        IDX[i] = list(range(sz))
        fsz[i] = int(sz/FOLDS)
        random.shuffle(IDX[i])
        IDX[i] = np.array(IDX[i])

        innersz = sz - fsz[i]
        innerIDX_0 = list(range(innersz))
        random.shuffle(innerIDX_0)
        innerIDX[i] = np.array(innerIDX_0)
        innerfsz[i] = int(innersz / innerFOLDS)

    if Validation == "nested_cv":
        bestHyperParsOut=[]
        print("---------- nested cv start ----------")
        for f in range(FOLDS):
            offset[i] = 0 + fsz[i]*f 
            # idx_test = IDX[offset:offset + fsz]
            # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
            idx_test = {}
            idx_train = {}
            for i in X.keys():
                idx_test[i] = IDX_validate_diff[i][[IDX[i][offset[i]:offset[i] + fsz[i]]]][0]
                idx_train[i] = IDX_validate_diff[i][[IDX[i][np.setdiff1d(np.arange(len(IDX[i])), np.arange(offset[i],offset[i] + fsz[i]))]]][0]

                # print(len(idx_test[i]) + len(idx_train[i]))
                
            print("Fold:",f)

            # innerdistance_X = distance_X[np.ix_(idx_train, idx_train)].copy()
            # innerdistance_Y = distance_Y[np.ix_(idx_train, idx_train)].copy()
    
            # setvar_besttune(innerFOLDS)
    
            setvar_tune(len(hyperparList), f = innerFOLDS)
            print("number of hyperpars combination: ", len(hyperparList))
            if hyperparfixed == False:
                for innerf in range(innerFOLDS):
                    # inneroffset = 0 + innerf*innerfsz
                    # idx_test_inner = innerIDX[inneroffset:inneroffset + innerfsz]
                    # idx_train_inner = innerIDX[np.array(np.setdiff1d(np.arange(len(idx_train)), np.arange(inneroffset,      inneroffset + innerfsz)))]
                    idx_test_inner = {}
                    idx_train_inner = {}
                    for i in X.keys():
                        inneroffset[i] = 0 + innerf*innerfsz[i]
                        idx_test_inner[i] = idx_train[i][[innerIDX[i][inneroffset[i]:inneroffset[i] + innerfsz[i]]]][0]
                        idx_train_inner[i] = idx_train[i][[innerIDX[i][np.array(np.setdiff1d(np.arange(len(idx_train[i])), np.arange(inneroffset[i],inneroffset[i] + innerfsz[i])))]]][0]
                        # print("-")
                        # print(featureMat_all[i][idx_test_inner[i],:].sum(1))
                    # print("first few training idx: ", np.sort(idx_train_inner[0:10]))
                    # print("first few testing idx: ", np.sort(idx_test_inner[0:10]))

                    # setDist(dist_X=distance_X_all[np.ix_(idx_train_inner, idx_train_inner)], dist_Y=distance_Y_all[np.ix_    (idx_train_inner, idx_train_inner)], dist_X_new=distance_X_all[np.ix_(idx_test_inner, idx_train_inner)])
                    # setDist()
        
                    print("Inner Fold:", innerf)
                    with parallel_backend('threading'):
                        Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix_all, idx_train_inner = idx_train_inner, idx_test_inner = idx_test_inner, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=innerf) for i in range(len(hyperparList)))
                    # setDist()
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
            results, pred = fold(idx_train,idx_test,featureMat_all,matrix_all,par=bestHyperPars)
            asgvar_cv(f=f, results=results)
    
        out = cv_results()
        features_new = {}
        for i in pred.keys():
            features_new[i] = pd.DataFrame(pred[i])
            features_new[i].index = drug
        return out, bestHyperParsOut, features_new
    
    elif Validation == "cv":
        print("---------- cv start ----------")
        # setvar_besttune(FOLDS)
    
        setvar_tune(len(hyperparList), f = FOLDS)
        if hyperparfixed == False:
                
            for f in range(FOLDS):
                offset[i] = 0 + fsz[i]*f 
                # idx_test = IDX[offset:offset + fsz]
                # idx_train = IDX[np.setdiff1d(np.arange(len(IDX)), np.arange(offset,offset + fsz))]
                idx_test = {}
                idx_train = {}
                for i in X.keys():
                    idx_test[i] = IDX_validate_diff[i][[IDX[i][offset[i]:offset[i] + fsz[i]]]][0]
                    idx_train[i] = IDX_validate_diff[i][[IDX[i][np.setdiff1d(np.arange(len(IDX[i])), np.arange(offset[i],offset[i] + fsz[i]))]]][0]
                # setDist(dist_X=distance_X_all[np.ix_(idx_train, idx_train)], dist_Y=distance_Y_all[np.ix_(idx_train, idx_train)], dist_X_new=distance_X_all[np.ix_(idx_test, idx_train)])
                # setDist()
                print("Fold:", f)
    
                with parallel_backend('threading'):
                    Parallel(n_jobs=n_jobs)(delayed(tuning_loop)(innermatrix = matrix_all, idx_train_inner = idx_train, idx_test_inner = idx_test, feature_matrix_inner = featureMat_all, hyperparList = hyperparList, i = i, f=f)for i in range(len(hyperparList)))
                # setDist()
            bestHyperPars, evalValue = tuning_results(tuneVar=hyperparList)
    
        else:
            # setDist()
            bestHyperPars = hyperparfixed
    
        # asg_besttune(f, value=evalValue, var=hyperpars)

        
        print("--- tuning end ---")
        # cv_results()
        # _, bestHyperPars = besttune()
        # validation
        idx_test = {}
        idx_train = {}
        for i in X.keys():
            idx_test[i] = IDX_validate[i]
            idx_train[i] = IDX_validate_diff[i]
        print('target size:', len(idx_test))
        print("------ best hyper pars: ", bestHyperPars, "------")
        results, pred = fold(idx_train,idx_test,featureMat_all,matrix_all,par=bestHyperPars)
        features_new = {}
        for i in pred.keys():
            features_new[i] = pd.DataFrame(pred[i])
            features_new[i].index = drug
        return results, bestHyperPars, features_new
    elif Validation == "completion":
        if hyperparfixed == False:
            raise ValueError("Please provide hyperparameter. hyperparfixed = ...")
        else:
            bestHyperPars = hyperparfixed
            idx_train = non_zero_idx_intersect_all
            idx_test = zero_idx_intersect_all
            pred = completionfold(idx_train,idx_test,featureMat_all,par=bestHyperPars)
            features_new = {}
            for i in pred.keys():
                features_new[i] = pd.DataFrame(pred[i])
                features_new[i].index = drug
            return features_new
