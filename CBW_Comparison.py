from sklearn.metrics import roc_curve

from sklearn.metrics import auc
import cloudpickle as pickle

import theano
from keras.callbacks import ModelCheckpoint  # , EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical

from graph_convolution import GraphConv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import h5py
import matplotlib.pyplot as plt
import os

fpr_keras1 = np.load("GCNN_fpr20.npy")
tpr_keras1= np.load("GCNN_tpr20.npy")
g_g_auc_keras1 = np.load("GCNN_auc20.npy")

fpr_keras2 = np.load("GCNN_fpr50.npy")
tpr_keras2= np.load("GCNN_tpr50.npy")
g_g_auc_keras2 = np.load("GCNN_auc50.npy")

fpr_keras3 = np.load("GCNN_fprEEGNETn.npy")
tpr_keras3= np.load("GCNN_tprEEGNETn.npy")
auc_keras3 = np.load("GCNN_aucEEGNETn.npy")

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras2, tpr_keras2, label='ChebNet (kernel size: 9; # of kernels: = 50) AUC = {:.3f}'.format(g_g_auc_keras2), color='g')
plt.plot(fpr_keras1, tpr_keras1, label='ChebNet (kernel size: 9; # of kernels: = 20) AUC = {:.3f}'.format(g_g_auc_keras1), color='b')
plt.plot(fpr_keras3, tpr_keras3, label='EEGNET AUC = {:.3f}'.format(auc_keras3), color='r')
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
# plt.title('ROC curve with Testing Set')
plt.legend(loc='best')
plt.show()