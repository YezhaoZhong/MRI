
import numpy as np
from scipy.linalg import inv
from scipy.spatial.distance import cdist
from itertools import product
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import numpy.linalg as LA
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import NMF
from scipy.linalg import svd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from sklearn.utils import parallel_backend
from sklearn.linear_model import LinearRegression
import time
import functools

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import KNN
from missingpy import MissForest

def setDist(dist_X=None, dist_Y=None, dist_X_new=None):
    global loaddist
    if (dist_X is None)|(dist_Y is None)|(dist_X_new is None):
        loaddist = False
    else:
        global distance_X
        global distance_Y
        global distance_X_new

        loaddist = True
        distance_X = dist_X
        distance_Y = dist_Y
        distance_X_new = dist_X_new

def runModels(Y,X,X_new,method_option,par=None):

    if method_option == "SKR":
        return SmoothedKR(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "KR":
        return KR(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "KRR":
        return KRR(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "MKRR":
        return KRR(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "Naive":
        n,_ = X_new.shape
        return Naive(Y=Y,n=n)
    elif method_option == "LNSM_RLN":
        return LNSM(Y=Y,X=X,X_new=X_new,par=par,distance_option="RLN")
    elif method_option == "LNSM_jaccard":
        return LNSM(Y=Y,X=X,X_new=X_new,par=par,distance_option="jaccard")
    elif method_option == "VKR":
        return VKR(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "SVM":
        return SVM(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "OCCA":
        return OCCA(Y=Y,X=X,X_new=X_new)
    elif method_option == "SCCA":
        return SCCA(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "RF":
        return RF(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "BRF":
        return BRF(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "TNMF":
        return TNMF(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "SRI":
        return SRI(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "MICE":
        return MICE(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "TKNN":
        return TKNN(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "TRF":
        return TRF(Y=Y,X=X,X_new=X_new,par=par)
    elif method_option == "TWNMF":
        return TWNMF(Y=Y,X=X,W=X_new,par=par)
    elif method_option == "BKR":
        return BKR(Y=Y,X=X,X_new=X_new,par=par)
    else:
        raise ValueError(f"{method_option} is not one of the models.")
    

def loadHyperpar(*hyperpars, method_option):
    if method_option == "SKR":
        n_par = 4
        print(f"The {method_option} requires hyperparameter lambda, c, sigma_X, sigma_Y")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "KR":
        n_par = 2
        print(f"The {method_option} requires hyperparameter lambda, sigma_X")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "KRR":
        n_par = 2
        print(f"The {method_option} requires hyperparameter lambda, sigma_X")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "MKRR":
        n_par = 3
        print(f"The {method_option} requires hyperparameter lambda, sigma_X")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "Naive":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. You do not need to load a hyperparameter list")
    elif method_option == "LNSM_RLN":
        n_par = 2
        print(f"The {method_option} requires hyperparameter alpha and lambda")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "LNSM_jaccard":
        n_par = 1
        print(f"The {method_option} requires hyperparameter alpha")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "VKR":
        n_par = 3
        print(f"The {method_option} requires hyperparameter lambda, sigma_X, k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "SVM":
        n_par = 3
        print(f"The {method_option} requires hyperparameter c, gamma")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "OCCA":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. You do not need to load a hyperparameter list")
    elif method_option == "SCCA":
        n_par = 2
        print(f"The {method_option} requires hyperparameter alpha")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "RF":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "BRF":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "TNMF":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "SRI":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "MICE":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "TKNN":
        n_par = 1
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "TRF":
        n_par = 0
        print(f"The {method_option} requires no hyperparameter")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "TWNMF":
        n_par = 2
        print(f"The {method_option} requires hyperparameter k")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    elif method_option == "BKR":
        n_par = 4
        print(f"The {method_option} requires hyperparameter lmd_x, lmd_y, sigma_x, sigma_y")
        if len(hyperpars) != n_par:
            raise ValueError(f"The number of hyperparameter is {n_par} when using {method_option}. Please reload the hyperparameter list.")
    else:
        raise ValueError(f"{method_option} is not one of the models.")
    
    hyperparList = list(product(*hyperpars))
    return hyperparList


def normalization(K):
    d1 = K.sum(axis=0) + 10e-8
    d2 = K.sum(axis=1) + 10e-8
    K_normalized = (K.T / np.sqrt(d2)).T / np.sqrt(d1)
    return K_normalized

# def smoother(Y, K, c):
#     return (1-c+c*normalization(K)).dot(1-c+c*normalization(K)).dot(Y)

# def SmoothedKR(Y,X,X_new,par):

#     lmd = par[0]
#     c = par[1]
#     sigma_X = par[2]
#     sigma_Y = par[3]
#     distance = cdist(X, X)**2
#     distance_new = cdist(X_new, X)**2

#     n,_ = X.shape # size of known drug

#     K = (np.exp(-distance/sigma_X**2))
#     K_new = (np.exp(-distance_new/sigma_X**2))
#     Lmd = np.diag(np.ones(n)*lmd)
#     distance_Y = cdist(Y, Y)**2
#     K_Y = (np.exp(-distance_Y/sigma_Y**2))
#     W = inv(K+Lmd).dot(smoother(Y, K_Y, c))
#     y_new = K_new.dot(W)
#     return y_new


# def matrix_to_tuple(matrix):
#     return tuple(map(tuple, matrix))

# @functools.lru_cache(maxsize=None)
# def expensive_inverse(K_X_tuple, Lmd_tuple):
#     K_X = np.array(K_X_tuple)
#     Lmd = np.array(Lmd_tuple)
#     result = inv(K_X.dot(K_X)+Lmd)  
#     return result

# def inverse(K_X, Lmd):
#     K_X_tuple = matrix_to_tuple(K_X)
#     Lmd_tuple = matrix_to_tuple(Lmd)
#     return expensive_inverse(K_X_tuple=K_X_tuple, Lmd_tuple=Lmd_tuple)

# @functools.lru_cache(maxsize=None)
# def expensive_smoother(Y_tuple, K_Y_tuple, c, n):
#     Y = np.array(Y_tuple)
#     K_Y = np.array(K_Y_tuple)
#     result = ((1-c)*np.diag(np.ones(n))+c*normalization(K_Y)).dot(((1-c)*np.diag(np.ones(n))+c*normalization(K_Y))).dot(Y)
#     return result

# def smoother2(Y, K_Y, c, n):
#     Y_tuple = matrix_to_tuple(Y)
#     K_Y_tuple = matrix_to_tuple(K_Y)
#     return expensive_smoother(Y_tuple=Y_tuple, K_Y_tuple=K_Y_tuple, c=c, n=n)


def smoother(Y, K_Y, c, n):
    Y_SS =  ((1-c)*np.diag(np.ones(n))+c/2*normalization(K_Y)).dot((1-c)*(np.diag(np.ones(n))+c/2*normalization(K_Y))).dot(Y)
    # return (np.diag(np.ones(n))+c*normalization(K)).dot(Y)
    # return (np.diag(np.ones(n))+c*normalization(K)).dot((np.diag(np.ones(n))+c*normalization(K))).dot(Y)
    min_val = np.min(Y_SS)
    max_val = np.max(Y_SS)
    min_max_normalized_matrix = (Y_SS - min_val) / (max_val - min_val)
    return min_max_normalized_matrix
def SmoothedKR(Y,X,X_new,par):

    lmd = par[0]
    c = par[1]
    sigma_X = par[2]
    sigma_Y = par[3]

    n,_ = X.shape # size of known drug
    Lmd = np.diag(np.ones(n)*lmd)
    if loaddist == True:
        K_X = (np.exp(-distance_X/sigma_X**2))
        K_new = (np.exp(-distance_X_new/sigma_X**2))
        K_Y = (np.exp(-distance_Y/sigma_Y**2))
    else:
        K_X = (np.exp(-cdist(X, X)**2/sigma_X**2))
        K_new = (np.exp(-cdist(X_new, X)**2/sigma_X**2))
        K_Y = (np.exp(-cdist(Y, Y)**2/sigma_Y**2))

    
    # W = inv(K+Lmd).dot((1-c)*Y+c*K.dot(Y))
    # W = inv(K+Lmd).dot(normalization(K_Y).dot(Y))
    # W = inv(K+c*normalization(K)+Lmd).dot(Y+c*normalization(K_Y).dot(Y))
    # W = inv(K.dot(K) + K_Y.dot(K_Y)+Lmd).dot((K + K_Y).dot(Y))
    # W = inv(K+c*normalization(K)+Lmd).dot(Y+c*smoother(Y, K_Y, 1))
    # W = inv(K_X+c*normalization(K_Y)+Lmd).dot(K_X).dot(smoother(Y, K_Y, c, n))
    # W = inv(K_X+c*normalization(K_Y)+Lmd).dot(Y+c*normalization(K_Y).dot(Y))
    # W = inv(K + Lmd).dot((Y))
    # W = inv(K+c*(K)+Lmd).dot(Y+c*normalization(K_Y).dot(Y))
    # t1 = time.time()

    # W = inverse(K_X,Lmd).dot(K_X).dot(smoother2(Y, K_Y, c, n))
    # t2 = time.time()

    # W = inv(K_X+Lmd).dot(smoother(Y, K_Y, c, n))
    W = inv(K_X.dot(K_X)+Lmd).dot(K_X).dot(smoother(Y, K_Y, c, n))
    # t3 = time.time()
    # print(t2-t1, t3-t2)

    y_new = K_new.dot(W)


    return y_new



# def normalization(K):
#     d1 = K.sum(axis=0) + 10e-8
#     d2 = K.sum(axis=1) + 10e-8
#     K_normalized = (K.T / cp.sqrt(d2)).T / cp.sqrt(d1)
#     return K_normalized

# def smoother(Y, K, c):
#     return (1-c)*(Y)+c*normalization(K).dot(Y)

# def SmoothedKR(Y,X,X_new,par):

#     lmd = par[0]
#     c = par[1]
#     sigma_X = par[2]
#     sigma_Y = par[3]

#     n,_ = X.shape # size of known drug
#     Lmd = np.diag(np.ones(n)*lmd)
#     if loaddist == True:
#         K_X = (np.exp(-distance_X/sigma_X**2))
#         K_new = (np.exp(-distance_X_new/sigma_X**2))
#         K_Y = (np.exp(-distance_Y/sigma_Y**2))
#     else:
#         K_X = (np.exp(-cdist(X, X)**2/sigma_X**2))
#         K_new = (np.exp(-cdist(X_new, X)**2/sigma_X**2))
#         K_Y = (np.exp(-cdist(Y, Y)**2/sigma_Y**2))
    
#     Y_cp = cp.array(Y)
#     K_Y_cp = cp.array(K_Y)
#     K_X_cp = cp.array(K_X)
#     K_new_cp = cp.array(K_new)
#     Lmd_cp = cp.array(Lmd)
    
#     W = cp.linalg.inv(K_X_cp+Lmd_cp).dot(smoother(smoother(Y_cp, K_Y_cp, c), K_Y_cp, c))
#     y_new_cp = K_new.dot(W)
#     y_new = cp.asnumpy(y_new_cp)
#     return y_new


def KR(Y,X,X_new,par):

    lmd = par[0]
    sigma_X = par[1]

    
    n,_ = X.shape # size of known drug
    Lmd = np.diag(np.ones(n)*lmd)
    if loaddist == True:
        K_X = (np.exp(-distance_X/sigma_X**2))
        K_new = (np.exp(-distance_X_new/sigma_X**2))
    else:
        K_X = (np.exp(-cdist(X, X)**2/sigma_X**2))
        K_new = (np.exp(-cdist(X_new, X)**2/sigma_X**2))

    W = inv(K_X+Lmd).dot(Y)
    y_new = K_new.dot(W)
    return y_new


def BKR(Y,X,X_new,par):

    lmd_X = par[0]
    lmd_Y = par[1]
    sigma_X = par[2]
    sigma_Y = par[3]

    
    n,_ = X.shape # size of known drug
    Lmd_X = np.diag(np.ones(n)*lmd_X)
    Lmd_Y = np.diag(np.ones(n)*lmd_Y)
    if loaddist == True:
        K_X = (np.exp(-distance_X/sigma_X**2))
        K_new = (np.exp(-distance_X_new/sigma_X**2))
        K_Y = (np.exp(-distance_Y/sigma_Y**2))
    else:
        K_X = (np.exp(-cdist(X, X)**2/sigma_X**2))
        K_new = (np.exp(-cdist(X_new, X)**2/sigma_X**2))
        K_Y = (np.exp(-cdist(Y, Y)**2/sigma_Y**2))

    W = inv(K_X+Lmd_X).dot(Y)

    yy_new = K_new.dot(W)
    YY = K_Y.dot(W)
    K_YY = (np.exp(-cdist(YY, YY)**2/sigma_Y**2))
    K_YY_new = (np.exp(-cdist(yy_new, YY)**2/sigma_Y**2))
    WW = inv(K_YY+Lmd_Y).dot(Y)
    y_new = K_YY_new.dot(WW)

    yy_new = K_new.dot(W)
    YY = K_Y.dot(W)
    # K_YY = (np.exp(-cdist(YY, YY)**2/sigma_Y**2))
    K_YY_new = (np.exp(-cdist(yy_new, YY)**2/sigma_Y**2))
    WW = inv(K_Y+Lmd_Y).dot(Y)
    y_new = K_YY_new.dot(WW)

    return y_new


        
def KRR(Y,X,X_new,par):

    lmd = par[0]
    sigma_X = par[1]

    n,_ = X.shape # size of known drug
    Lmd = np.diag(np.ones(n)*lmd)
    if loaddist == True:
        K_X = (np.exp(-distance_X/sigma_X**2))
        K_new = (np.exp(-distance_X_new/sigma_X**2))
    else:
        K_X = (np.exp(-cdist(X, X)**2/sigma_X**2))
        K_new = (np.exp(-cdist(X_new, X)**2/sigma_X**2))

    W = inv(K_X.dot(K_X)+Lmd).dot(K_X.dot(Y))
    y_new = K_new.dot(W)
    return y_new

def MKRR(Y,X,X_new,par):


    lmd = par[0]
    sigma_X = par[1]
    c = par[2]
    K_X = 0
    K_new = 0
    p = c
    for i in X.keys():

        n,_ = X[i].shape # size of known drug
        Lmd = np.diag(np.ones(n)*lmd)
    # if loaddist == True:
    #     K_X = (np.exp(-distance_X/sigma_X**2))
    #     K_new = (np.exp(-distance_X_new/sigma_X**2))
    # else:
        K_X += (1-p)*(np.exp(-cdist(X[i], X[i])**2/sigma_X**2))
        K_new += (1-p)*(np.exp(-cdist(X_new[i], X[i])**2/sigma_X**2))
        p = 1-p

    W = inv(K_X.dot(K_X)+Lmd).dot(K_X.dot(Y))
    y_new = K_new.dot(W)
    return y_new

def Naive(Y,n):
    _,m = Y.shape
    mean_side_effect_score = (Y.copy()).mean(axis=0)
    y_new = np.zeros((n, m))
    # Set the prediction into mean
    for i in range(m):
        y_new[:, i] =  mean_side_effect_score[i]
    return y_new

def Jaccard(X,X_new):
    W = 1 - cdist(X, X, "jaccard")
    W_new = 1 - cdist(X_new, X, "jaccard")
    return W, W_new

def RLN(X,X_new,lmd):
    n, _ = X.shape
    m, _ = X_new.shape

    neigh = NearestNeighbors(n_neighbors = 200)
    neigh.fit(X)
    W = np.zeros((n, n))
    W_new = np.zeros((m, n))
    clf = Ridge(alpha=lmd)

    N = neigh.kneighbors(X, 200, return_distance=False)
    for i in range(n):
        # print("test")
        X_knn = X[N[i], :]
        clf.fit(X_knn.T, X[i, :])
        W[i, N] = clf.coef_

    N_new = neigh.kneighbors(X_new, 200, return_distance=False)
    for i in range(m):
        X_knn_new = X[N_new[i], :]
        clf.fit(X_knn_new.T, X_new[i, :])
        W_new[i, N_new[i]] = clf.coef_

    return W, W_new

def LNSM(Y,X,X_new,par,distance_option):
    alpha = par[0]
    Y_0 = (Y.copy())
    if distance_option == "RLN":
        lmd = par[1]
        W, W_new = RLN(X=X, X_new=X_new, lmd=lmd)
    elif distance_option == "jaccard":
        W, W_new = Jaccard(X=X, X_new=X_new)

    n, _ = W.shape
    
    Y = (1-alpha)*inv(np.diag(np.ones(n)) - alpha*W).dot(Y_0)
    # cost_t1 = alpha * np.trace(np.dot(np.dot(Y_t1.T, 1 - W), Y_t1)) + (1 - alpha)*LA.norm(Y_t1 - Y_0)**2

    # for i in range(max_iter):
    #     Y_t2 = alpha * np.dot(W, Y_t1) + (1 - alpha) * Y_0

    #     cost_t2 = alpha * np.trace(np.dot(np.dot(Y_t2.T, 1 - W), Y_t2)) + (1 - alpha)*LA.norm(Y_t2 - Y_0)**2

    #     Y_t1 = Y_t2.copy()
    #     cost_t1 = cost_t2

    #     if (cost_t2 - cost_t1) < (cost_t1 / 100):
    #         # print("LNSM converged")
    #         break
    #     if i == (max_iter - 1):
    #         print("maximum iteration reached")

    # Y = Y_t2.copy()

    # Y = (1 - alpha) * np.dot(np.linalg.pinv(1 - alpha * W), Y_0)
    y_new = np.dot(W_new, Y)
    return y_new


def TNMF(Y,X,X_new,par):
    k = par[0]
    nmf_model = NMF(n_components=k, random_state=1949, max_iter=10000)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    WH = np.dot(W, H)
    # _, _, reconWH = GNMF(bipart_graph=WH.T, component=k, WMK=Similarity, lmd=lmd, max_iter=10000, tolerance=1/10000)
    # y_new = reconWH.T[0:n, 0:m]
    return WH

def SRI(Y,X,X_new,par):
    imputer = SimpleImputer(strategy='mean')  # 'median', 'most_frequent', or 'constant'
    data_imputed = imputer.fit_transform(X_new)
    return data_imputed


def MICE(Y,X,X_new,par):
    iter_imputer = IterativeImputer(max_iter=10, random_state=1949)
    data_imputed_iterative = pd.DataFrame(iter_imputer.fit_transform(X_new))
    return data_imputed_iterative

def TKNN(Y,X,X_new,par):
    k = par[0]
    data_filled_knn = KNN(k=k).fit_transform(X_new)
    return data_filled_knn

def TRF(Y,X,X_new,par):
    imputer = MissForest()
    data_imputed = imputer.fit_transform(X_new)
    return data_imputed

def GNMF(bipart_graph, component, WMK, lmd, max_iter, tolerance=1/1000000):
    np.random.seed(1949)

    W = WMK.copy()
    X = bipart_graph.copy()
    m, n = X.shape
    k = component
 
    D = np.matrix(np.diag(np.asarray(W.copy()).sum(axis=1)))
    L = D.copy() - W.copy()

    # Initialize U & V

    U = np.random.random((m, k))
    V = np.random.random((n, k))

    # Updating U V
    eps = 2**-8

    term1 = LA.norm(X - np.dot(U, V.T))**2
    term2 = lmd * np.trace(np.dot(np.dot(V.T, L), V))
    Obj0 = term1 + term2
    Obj1 = Obj0


    for i in range(max_iter):
        XV = np.dot(X, V)
        UVtV = np.dot(np.dot(U, V.T), V) + eps

        U *= XV
        U /= UVtV
        
        XtU_lmdWV = np.dot(X.T, U) + lmd*np.dot(W, V)
        VUtU_lmdDV = np.dot(np.dot(V, U.T), U) + lmd*np.dot(D, V) + eps
        V *= XtU_lmdWV
        V /= VUtU_lmdDV

        # Objective function
        
        term1 = LA.norm(X - np.dot(U, V.T))**2
        term2 = lmd * np.trace(np.dot(np.dot(V.T, L), V))
        Obj2 = term1 + term2    
        ObjDiff = Obj1 - Obj2
        Obj1 = Obj2

        if(ObjDiff < (Obj0 *tolerance)):
            # print("Converged in iteration: ", i, "ObjDiff: ", ObjDiff, "Obj: ", Obj2)
            return(U, V, np.dot(U, V.T))
        elif i == max_iter - 1:
            print("Has not converged, reach the maximum iteration")
            return(U, V, np.dot(U, V.T))
    




def VKR(Y,X,X_new,par):
    sigma = par[0]
    lmd = par[1]
    k = par[2]
    # V,U,preds = NMF(Y, component=k, max_iter=1000)
    nmf_model = NMF(n_components=k, random_state=1949, max_iter=1000)
    V = nmf_model.fit_transform(Y)
    U = nmf_model.components_

    Vpreds = KRR(V,X,X_new,par=[lmd, sigma])
    y_new = Vpreds.dot(U)
    
    return y_new

def SVMloop(idx, Y, X, X_new, par):
    c = par[0]
    g = par[1]
    y_i = np.array(Y[:, idx].copy()).tolist()
    if len(np.unique(y_i)) == 1:
        y_new_SVM[:, idx] = np.unique(y_i)
    else:
        svr = SVR(kernel="rbf", C = c, gamma = g).fit(X, y_i)
        y_new_SVM[:, idx] = svr.predict(X_new)


def SVM(Y,X,X_new,par):
    X_normalized = (X.copy() - X.mean(axis=0)) / (X.std(axis=0) + 10e-8)
    m,_ = X_new.shape
    _,n = Y.shape
    global y_new_SVM
    y_new_SVM = np.zeros((m, n)).astype(float)
    
    with parallel_backend('threading'):
        Parallel(n_jobs=1)(delayed(SVMloop)(idx = i, Y=Y, X=X_normalized, X_new=X_new,par=par) for i in range(n))

    return y_new_SVM

# class OCCAf:
#     def __init__(self, n_components=2):
#         self.n_components = n_components
    
#     def fit(self, X, Y):
#         # Center the data
#         X_mean = np.mean(X, axis=0)
#         Y_mean = np.mean(Y, axis=0)
#         X_centered = X - X_mean
#         Y_centered = Y - Y_mean
        
#         # Singular Value Decomposition (SVD)
#         U, S, V = np.linalg.svd(np.dot(X_centered.T, Y_centered), full_matrices=False)
        
#         # Select top n_components
#         self.X_c = U[:, :self.n_components]
#         self.Y_c = V.T[:, :self.n_components]
        
#         # Store means for later use
#         self.X_mean = X_mean
#         self.Y_mean = Y_mean
    
#     def transform(self, X, Y):
#         # Center the data
#         X_centered = X - self.X_mean
#         Y_centered = Y - self.Y_mean
        
#         # Project onto canonical vectors
#         X_c_transformed = np.dot(X_centered, self.X_c)
#         Y_c_transformed = np.dot(Y_centered, self.Y_c)
        
#         return X_c_transformed, Y_c_transformed

#     def transform_X(self, X):
#         X_centered = X - self.X_mean
#         return np.dot(X_centered, self.X_c)

#     def transform_Y(self, Y):
#         Y_centered = Y - self.Y_mean
#         return np.dot(Y_centered, self.Y_c)


class OCCAf:
    def __init__(self, n_components=2):
        self.n_components = n_components
    
    def fit(self, X, Y):
        # Center the data
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        
        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(np.dot(X_centered.T, Y_centered), full_matrices=False)
        
        # Select top n_components
        self.X_c = U[:, :self.n_components]
        self.Y_c = Vt.T[:, :self.n_components]
        
        # Store means for later use
        self.X_mean = X_mean
        self.Y_mean = Y_mean
    
    def transform(self, X, Y):
        # Center the data
        X_centered = X - self.X_mean
        Y_centered = Y - self.Y_mean
        
        # Project onto canonical vectors
        X_c_transformed = np.dot(X_centered, self.X_c)
        Y_c_transformed = np.dot(Y_centered, self.Y_c)
        
        return X_c_transformed, Y_c_transformed

    def transform_X(self, X):
        X_centered = X - self.X_mean
        return np.dot(X_centered, self.X_c)

    def transform_Y(self, Y):
        Y_centered = Y - self.Y_mean
        return np.dot(Y_centered, self.Y_c)




    
def OCCA(Y,X,X_new):

    # Fit the OCCA model
    _, n = Y.shape
    print(Y.shape)
    print(X.shape)
    print(X_new.shape)
    occa_model = OCCAf(n_components=n)
    occa_model.fit(X, Y)
    # Transform training data
    X_train_c, Y_train_c = occa_model.transform(X, Y)
    
    # Train a regression model in the canonical space
    regressor = LinearRegression()
    regressor.fit(X_train_c, Y_train_c)
    print(X_new.shape)
    # Transform new data
    X_new_c = occa_model.transform_X(X_new)
    print(X_new_c.shape)
    # Make predictions in the canonical space
    y_new = regressor.predict(X_new_c)
    # print(y_new.shape)

    return y_new


class SCCAf:
    def __init__(self, n_components=2, alpha=1.0, max_iter=1000, tol=1e-12):
        self.n_components = n_components  # Number of canonical components
        self.alpha = alpha  # Regularization parameter
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = tol  # Tolerance for convergence
        
    def fit(self, X, Y):
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        n, p = X.shape
        m, q = Y.shape
        
        # Initialize canonical vectors
        U = np.random.randn(p, self.n_components)
        V = np.random.randn(q, self.n_components)
        
        for _ in range(self.max_iter):
            # Update U
            U_new = self.update(U, V, X, Y, alpha=self.alpha)
            
            # Update V
            V_new = self.update(V, U_new, Y, X, alpha=self.alpha)
            
            # Check convergence
            if np.linalg.norm(U_new - U) < self.tol and np.linalg.norm(V_new - V) < self.tol:
                break
            
            U = U_new
            V = V_new
        
        self.U = U
        self.V = V
        self.X_c = U[:, :self.n_components]
        self.Y_c = V[:, :self.n_components]
        self.X_mean = X_mean
        self.Y_mean = Y_mean
        
    def update(self, W, Z, X, Y, alpha):
        # Compute cross-covariance matrix
        C = np.dot(X.T, Y)
        
        # Regularization matrix
        D = np.diag(np.sum(W**2, axis=0) + alpha)
        
        # Compute new canonical vectors
        _, _, V = svd(np.dot(np.dot(C, Z), np.linalg.inv(np.dot(np.dot(Z.T, Z), D))))
        
        return np.dot(np.dot(C.T, W), V)
    
    def transform(self, X, Y):
        # Center the data
        X_centered = X - self.X_mean
        Y_centered = Y - self.Y_mean
        
        # Project onto canonical vectors
        X_c_transformed = np.dot(X_centered, self.X_c)
        Y_c_transformed = np.dot(Y_centered, self.Y_c)
        
        return X_c_transformed, Y_c_transformed

    def transform_X(self, X):
        X_centered = X - self.X_mean
        return np.dot(X_centered, self.X_c)

    def transform_Y(self, Y):
        Y_centered = Y - self.Y_mean
        return np.dot(Y_centered, self.Y_c)

def SCCA(Y,X,X_new,par):
    a = par[0]
    _, n = Y.shape
    # Fit the OCCA model
    scca_model = SCCAf(n_components=n, alpha=a)
    scca_model.fit(X, Y)
    # Transform training data
    X_train_c, Y_train_c = scca_model.transform(X, Y)
    
    # Train a regression model in the canonical space
    regressor = LinearRegression()
    regressor.fit(X_train_c, Y_train_c)
    
    # Transform new data
    X_new_c = scca_model.transform_X(X_new)
    
    # Make predictions in the canonical space
    y_new = regressor.predict(X_new_c)
    return y_new


def RF(Y,X,X_new,par):

    # m,_ = X_new.shape
    # _,n = Y.shape
    # global y_new_RF
    k = par[0]
    rf = RandomForestRegressor(n_estimators=k, random_state=1949).fit(X, Y)
    y_new_RF = rf.predict(X_new)
    # y_new_RF = np.zeros((m, n)).astype(float)
    
    # with parallel_backend('threading'):
    #     Parallel(n_jobs=1)(delayed(RFloop)(idx = i, Y=Y, X=X, X_new=X_new,par=par) for i in range(n))
    return y_new_RF

def BRF(Y,X,X_new,par):
    iter = 50
    lr = 1/iter
    y_new = RF(Y,X,X_new,par=par)
    for i in range(iter-1):
        y_new += lr*RF(Y,X,X_new,par=par)
    return y_new

def TWNMF(Y,X,W,par):
    k = par[0]
    _, _, UV = WNMF(bipart_graph=X, W=W, component=k, max_iter=100000, tolerance=1/100000)
    return UV

def WNMF(bipart_graph, W, component, max_iter, tolerance=1/10000):
    np.random.seed(1949)

    X = bipart_graph.copy()
    m, n = X.shape
    k = component

    # Initialize U & V

    U = np.random.random((m, k))
    V = np.random.random((n, k))

    # Updating U V
    eps = 2**-8
    Obj0 = 0.5*LA.norm(W*(X - np.dot(U, V.T)))**2
    Obj1 = Obj0

    for i in range(max_iter):
        XV = np.dot(W*X, V)
        UVtV = np.dot(W*np.dot(U, V.T), V) + eps
        U *= XV
        U /= UVtV
        XtU_lmdWV = np.dot(W.T*X.T, U)
        VUtU_lmdDV = np.dot(W.T*np.dot(V, U.T), U) + eps
        V *= XtU_lmdWV
        V /= VUtU_lmdDV
        # Objective function
        Obj2 = 0.5*LA.norm(W*(X - np.dot(U, V.T)))**2
        ObjDiff = Obj1 - Obj2
        Obj1 = Obj2
        if(ObjDiff < (Obj0 *tolerance)):
            # print("Converged in iteration: ", i, "ObjDiff: ", ObjDiff, "Obj: ", Obj2)
            return(U, V, np.dot(U, V.T))
        elif i == max_iter - 1:
            print("Has not converged, reach the maximum iteration")
            return(U, V, np.dot(U, V.T))




# def TNMF(Y,X,X_new,par):
#     k = par[0]
#     _, _, UV = NMF2(bipart_graph=X, W=X_new, component=k, max_iter=100000, tolerance=1/100000)
#     return UV

# def NMF2(bipart_graph, W, component, max_iter, tolerance=1/10000):
#     np.random.seed(1949)

#     X = bipart_graph.copy()
#     m, n = X.shape
#     k = component

#     # Initialize U & V

#     U = np.random.random((m, k))
#     V = np.random.random((n, k))

#     # Updating U V
#     eps = 2**-8
#     Obj0 = 0.5*LA.norm(X - np.dot(U, V.T))**2
#     Obj1 = Obj0

#     for i in range(max_iter):
#         XV = np.dot(X, V)
#         UVtV = np.dot(np.dot(U, V.T), V) + eps
#         U *= XV
#         U /= UVtV
#         XtU_lmdWV = np.dot(X.T, U)
#         VUtU_lmdDV = np.dot(np.dot(V, U.T), U) + eps
#         V *= XtU_lmdWV
#         V /= VUtU_lmdDV
#         # Objective function
#         Obj2 = 0.5*LA.norm((X - np.dot(U, V.T)))**2
#         ObjDiff = Obj1 - Obj2
#         Obj1 = Obj2
#         if(ObjDiff < (Obj0 *tolerance)):
#             # print("Converged in iteration: ", i, "ObjDiff: ", ObjDiff, "Obj: ", Obj2)
#             return(U, V, np.dot(U, V.T))
#         elif i == max_iter - 1:
#             print("Has not converged, reach the maximum iteration")
#             return(U, V, np.dot(U, V.T))


# def SVM(Y,X,X_new,par):
#    c = par[0]
#    g = par[1]
#    svr = SVR(kernel='rbf', gamma=g, C=c)
#    pipeline = make_pipeline(StandardScaler(), MultiOutputRegressor(svr), n_jobs=1)
#    pipeline.fit(X, Y)
#    y_new = pipeline.predict(X_new)
#    return y_new