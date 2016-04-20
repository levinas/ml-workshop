#! /usr/bin/env python

import argparse
import errno
import h5py
import os
import sys

import numpy as np
import numpy.random

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve


numpy.set_printoptions(threshold=numpy.nan)

def read_dataset(fname):
    f = open(fname)
    cols = f.readline().split(",")
    skipcols = 2  # first 3 columns (CellLine PubChemID ZScore) followed by features
    X_label = cols[skipcols:]
    X = np.loadtxt(f, delimiter=",", unpack=True, usecols=range(skipcols, len(cols)))
    X = np.transpose(X)
    y = np.genfromtxt(fname, skip_header=1, delimiter=",", usecols=[skipcols-1])
    median = np.median(y)
    y = np.transpose(map(lambda x: 1 if x > median else 0, y))
    return X, y, X_label

def test():
    # X, y, labels = read_dataset('sample1/s1.d200.z1')
    X, y, labels = read_dataset('s1.toy')
    print X
    print y
    print labels

def score_format(metric, score, eol='\n'):
    return '{:<15} = {:.5f}'.format(metric, score) + eol

def top_important_features(clf, feature_names, num_features=20):
    if not hasattr(clf, "feature_importances_"):
        return
    fi = clf.feature_importances_
    features = [ (f, n) for f, n in zip(fi, feature_names)]
    top = sorted(features, key=lambda f:f[0], reverse=True)[:num_features]
    return top

def sprint_features(top_features, num_features=20):
    str = ''
    for i, feature in enumerate(top_features):
        if i >= num_features: return
        str += '{}\t{:.5f}\n'.format(feature[1], feature[0])
    return str

def make_caffe_files(X, y, path):

    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path): pass
        else: raise

    sss = StratifiedShuffleSplit(y, 1, test_size=0.1)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_h5 = os.path.join(path, 'train.h5')
        test_h5 = os.path.join(path, 'test.h5')

        train_filename = os.path.join(path, 'train.txt')
        test_filename = os.path.join(path, 'test.txt')

        with h5py.File(train_h5, 'w') as f:
            f['data'] = X_train
            f['label'] = y_train.astype(np.float32)

        with h5py.File(test_h5, 'w') as f:
            f['data'] = X_test
            f['label'] = y_test.astype(np.float32)

        with open(train_filename, 'w') as f:
            f.write('train.h5\n')

        with open(test_filename, 'w') as f:
            f.write('test.h5\n')

        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--caffe', action='store_true', help='save train and test files in HDF5 for Caffe')
    parser.add_argument('-o', '--outdir', action='store', help='store results files to a specified directory')
    parser.add_argument('fname', default='s1.toy', help='drug data file (columns: [CellLine PubChemID ZScore Feature1 Feature2 ...])')
    args = parser.parse_args()

    X, y, labels = read_dataset(args.fname)
    prefix = os.path.basename(args.fname)
    if args.outdir:
        prefix = os.path.join(args.outdir, prefix)

    if args.caffe:
        make_caffe_files(X, y, prefix+'.caffe')

    classifiers = [
                    ('RF',  RandomForestClassifier(n_estimators=100, n_jobs=10)),
                    ('LogRegL1', LogisticRegression(penalty='l1')),
                    # ('SVM', SVC()),
                    # ('Ada', AdaBoostClassifier(n_estimators=100)),
                    # ('KNN', KNeighborsClassifier()),
                  ]

    for name, clf in classifiers:
        sys.stderr.write("> {}\n".format(name))
        train_scores, test_scores = [], []
        probas = None
        tests = None
        preds = None

        skf = StratifiedKFold(y, n_folds=10, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf):
            # print("  > TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            train_scores.append(clf.score(X_train, y_train))
            test_scores.append(clf.score(X_test, y_test))
            sys.stderr.write("  fold #{}: score={:.3f}\n".format(i, clf.score(X_test, y_test)))

            y_pred = clf.predict(X_test)
            preds = np.concatenate((preds, y_pred)) if preds is not None else y_pred
            tests = np.concatenate((tests, y_test)) if tests is not None else y_test

            if hasattr(clf, "predict_proba"):
                probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
                probas = np.concatenate((probas, probas_)) if probas is not None else probas_

        roc_auc_core = None
        if probas is not None:
            fpr, tpr, thresholds = roc_curve(tests, probas[:, 1])
            roc_auc_score = auc(fpr, tpr)
            roc_fname = "{}.{}.ROC".format(prefix, name)
            with open(roc_fname, "w") as roc_file:
                roc_file.write('\t'.join(['Threshold', 'FPR', 'TPR'])+'\n')
                for ent in zip(thresholds, fpr, tpr):
                    roc_file.write('\t'.join("{0:.5f}".format(x) for x in list(ent))+'\n')

        scores_fname = "{}.{}.scores".format(prefix, name)
        ms = 'accuracy_score f1_score precision_score recall_score log_loss'.split()
        with open(scores_fname, "w") as scores_file:
            for m in ms:
                s = getattr(metrics, m)(tests, preds)
                scores_file.write(score_format(m, s))
            avg_train_score = np.mean(train_scores)
            avg_test_score = np.mean(test_scores)
            if roc_auc_score is not None:
                scores_file.write(score_format('roc_auc_score', roc_auc_score))
            scores_file.write(score_format('avg_test_score', avg_test_score))
            scores_file.write(score_format('avg_train_score', avg_train_score))
            scores_file.write('\nModel:\n{}\n\n'.format(clf))

        top_features = top_important_features(clf, labels)
        if top_features is not None:
            fea_fname = "{}.{}.features".format(prefix, name)
            with open(fea_fname, "w") as fea_file:
                fea_file.write(sprint_features(top_features))

        sys.stderr.write('  test={:.5f} train={:.5f}\n\n'.format(avg_test_score, avg_train_score))


if __name__ == '__main__':
    # test()
    main()
