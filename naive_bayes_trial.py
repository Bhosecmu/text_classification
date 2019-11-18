import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
plt.rcParams['figure.figsize']=(10,6)
import glob
from string import punctuation
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import operator
from scipy.sparse import csr_matrix

path_categories_list = "C:\\Users\\athar\\Downloads\\ML_project\\20_newsgroups\\*"
categories_list = glob.glob(path_categories_list)

print(1, 2)
for i in range(20):
    temppath = str(categories_list[i])
    str1 = temppath + "\\*"
    file_list = glob.glob(str1)
    path_list_300 =  file_list[700:1000]
# syntax for sparse matrix input
row_nonzero = np.array([0, 1, 2, 3, 3, 4, 5]) # row locations for nonzero elements
col_nonzero = np.array([0, 1, 0, 0, 1, 1, 0]) # col locations for nonzero elements
nonzero_element = np.array([4, 4, 2, 6, 6, 6, 8]) # nonzero element value

# create a sparse matrix
X = csr_matrix((nonzero_element, (row_nonzero, col_nonzero)),shape=(6, 2))#.toarray()
Y = np.array([1, 4, 4, 2, 2, 3])
#print(X, Y)

x_train = []
x_train.append(row_nonzero)
#print(x_train)

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X, Y)
print(classif.predict([[0, 4]]))
#clf = GaussianNB()
clf = MultinomialNB()#alpha = 0.01)
print('here')
clf.fit(X, Y)


print(clf.predict([[0, 4]]))

# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y))

# print(clf_pf.predict([[0, 4]]))

# X = np.array([[-1, 0], [0, -1], [-3, 0], [1, 1], [0, 1], [3, 0]])
# print(X)
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
