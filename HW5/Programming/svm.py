# -*- coding: utf-8 -*-
"""SVM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FWm7WZw3fQkC-99NgBGmPAnoJPZkwEYe
"""

!gdown 1sCTQXp9kTcm8_jckqC-v_gWxIsFCcUbS
!gdown 1vK24A09o5Nev5qj1qNhndFe6beTWSDRU
!gdown 1THvOuf_EOn6c_6TLy0Bqs23BP2NraBR2

import numpy as np
import pandas as pd
import csv
from sklearn.svm import SVC

def normalize(X, mu_x=None, std_x=None):
  pass
  
  return X

def load_train():
  X = pd.read_csv("X_train")  # 'age', 'fnlwgt', 'hours_per_week', 'capital_gain', 'capital_loss' are coninuous, others are discrete
  Y = pd.read_csv("Y_train", header = None).values.reshape(-1)
  X = normalize(X)

  return X, Y

def load_test():
  X = pd.read_csv("X_test")
  X = normalize(X)
  return X

train_X, train_Y= load_train()
test_X = load_test()

clf = SVC(kernel='linear', random_state=0)
clf.fit(train_X, train_Y)

results = clf.predict(test_X)
with open('predict.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id','label'])
    for i, x in enumerate(results):
      writer.writerow([i + 1, int(x)])