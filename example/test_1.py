import tensorflow as tf
from sklearn.metrics import roc_auc_score
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser

sys.path.append("..")
from DeepFM import DeepFM


def _load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    cols = [c for c in dfTrain.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["target"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    for i, (train_idx, valid_idx) in enumerate(folds):
        # k折交叉，每一折中的fit中，含有epoch轮训练，每一次epoch拆分了batch来喂入
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)  # fit中包含对train和valid的评估

        yy = dfm.predict(Xi_valid_, Xv_valid_)
        # print("type(yy):",type(yy))
        # print("type(y_valid_):", type(y_valid_))

        # print("yy.shape:",yy.shape)               #yy : array
        # print("y_valid_.shape:", y_valid_.shape)  #y_valid_ : list

        #print("yy:", yy)  # 原始的predict出来的是概率值
        for index in range(len(yy)):
            if (yy[index] <= 0.5):
                yy[index] = 0
            else:
                yy[index] = 1

        #print("y_valid_:", y_valid_)

        print("accuracy_score(y_valid_, yy):", accuracy_score(y_valid_, yy))

        y_train_meta[valid_idx, 0] = yy

        y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)

    y_test_meta /= float(len(folds))

    return y_train_meta, y_test_meta


# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 10,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2017
}

dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

print("over")

# Xi_train, Xv_train, y_train = prepare(...)
# Xi_valid, Xv_valid, y_valid = prepare(...)