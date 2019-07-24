# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import argparse
import pickle
import os
import numpy as np
from time import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=False, default='output/embeddings_train.pickle',
	help="path to serialized db of facial embeddings")
ap.add_argument("-E", "--Embeddings", required=False, default='output/embeddings_test.pickle',
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=False, default= 'output/recognizer2.pickle',
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=False, default= 'output/le.pickle',
	help="path to output label encoder")
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
data_train = pickle.loads(open(args["embeddings"], "rb").read())
data_test = pickle.loads(open(args["Embeddings"], "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()

# Loading data
X_train = data_train["embeddings"]
X_test = data_test["embeddings"]
y_train = le.fit_transform(data_train["names"])
y_test = le.fit_transform(data_test["names"])

# No need for spliting the data
# # splite the data into train and test
# X_train, X_test, y_train, y_test = train_test_split(
#     data["embeddings"], labels,stratify=labels, test_size=0.2)

# param_grid = {'tol':[1e-4,1e-5,1e-6]}
# clf_2 =  GridSearchCV(LogisticRegression(solver='liblinear',multi_class='multinomial',class_weight='balanced'), 
# 					param_grid, cv=5, iid=False)

# clf_2 = clf_2.fit(X_train, y_train)

# y_pred = clf_2.predict(X_test)
# print(classification_report(y_test, y_pred, target_names=os.listdir("dataset")))
# print(confusion_matrix(y_test, y_pred, labels=range(len(os.listdir("dataset")))))


print("Fitting the classifier to the training set")
t0 = time()

# The value that grid search will go over
param_grid = {'learning_rate_init':[1e-5,5e-5,1e-4,0.001,0.005],
             'alpha':[1e-5,1e-6,1e-7],
             'epsilon':[1e-7,1e-8,1e-9],
             'tol':[1e-2,1e-3]
             }

# Setting the parmeter for the training
mlp = GridSearchCV(MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32), learning_rate='adaptive'),
                   param_grid, cv=10, iid=False)

# Start training
mlp.fit(X_train, y_train) 


print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(mlp.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = mlp.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred, target_names=os.listdir("dataset")))
print(confusion_matrix(y_test, y_pred, labels=range(len(os.listdir("dataset")))))

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(mlp))
f.close()

# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()