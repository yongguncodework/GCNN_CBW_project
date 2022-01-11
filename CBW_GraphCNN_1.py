'''
MNIST experiment comparing the performance of the graph convolution
with Logistic regression and fully connected neural networks.

The data graph structured is learned from the correlation matrix.

This reproduce the results shown in table 2 of:
"A generalization of Convolutional Neural Networks to Graph-Structured Data"
'''

### Dependencies
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

np.random.seed(2017) # for reproducibility

### Parameters
batch_size=128
epochs=100
num_neighbors= 8
filters = 50
num_classes = 2
results = dict()

### Load the data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()



data_file_name = 'CBW_freqband1.mat'
DATA_FOLDER_PATH = '/media/eeglab/YG_Storage/cyberwell/CBW_Data/'

FILE_PATH = DATA_FOLDER_PATH + '/' + data_file_name

# mat_2 = scipy.io.loadmat(FILE_PATH)
mat_2 = h5py.File(FILE_PATH)
mat_2.keys()

data_x = mat_2['data_reform']
data_y = mat_2['label']
cormat = mat_2['cormat_pos']
cormat2 = mat_2['cormat_abs']
cormat3 = mat_2['cormat_neg']
# data_y = data_y[:, 1]
data_y = to_categorical(data_y, 2)
# X_train = np.transpose(batch_images, (0, 3, 1, 2))
# data_x = np.transpose(data_x, (1, 0))
# data_y = np.transpose(data_y, (1, 0))
# data_x = data_x[:, 0:6, :]
# data_x = np.expand_dims(data_x, axis=1)

X_orig = np.array(data_x)
Y_orig = np.array(data_y)
A_orig = np.array(cormat)
A_orig2 = np.array(cormat2)
A_orig3 = np.array(cormat3)




X_train, X_test, Y_train, Y_test = train_test_split(X_orig, Y_orig, test_size=0.20, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

ix = np.where(np.var(X_train,axis=0)>0)[0]
X_train = X_train[:,ix]
X_test = X_test[:,ix]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
# Y_train = to_categorical(y_train, num_classes)
# Y_test = to_categorical(y_test, num_classes)

### Prepare the Graph Correlation matrix
# corr_mat = np.array(normalize(np.abs(np.corrcoef(X_train.transpose())),
#                               norm='l1', axis=1),dtype='float64') #i might use later YG

corr_mat2 = A_orig
corr_mat = A_orig2
corr_mat3 = A_orig3

corr_mat_mix = corr_mat*corr_mat2*corr_mat3
# corr_mat = np.ones([160, 160])
graph_mat = np.argsort(corr_mat,1)[:,-num_neighbors:]
graph_mat2 = np.argsort(corr_mat2,1)[:,-num_neighbors:]
graph_mat3 = np.argsort(corr_mat3,1)[:,-num_neighbors:]

# %%

## Single layer of Graph Convolution
g_model = Sequential()
g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
                      num_neighbors=num_neighbors, activation='relu',
                      input_shape=(X_train.shape[1],1,)))
g_model.add(Dropout(0.2))
g_model.add(Flatten())
g_model.add(Dense(2, activation='softmax'))

g_model.summary()

g_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
                metrics=['accuracy'])

dataset_name = 'EEGNET_Cyber'
os.makedirs('models/%s' % dataset_name, exist_ok=True)
checkpointPath = '.models'
checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=1, save_best_only=True)

aaa = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
bbb = (X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test)
results['g'] = g_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_error = 1-results['g'].__dict__['history']['val_acc'][-1]


# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(results['g'].history['loss'], 'y', label='train loss')
# loss_ax.plot(results['g'].history['val_loss'], 'r', label='val loss')
#
# acc_ax.plot(results['g'].history['acc'], 'b', label='train acc')
# acc_ax.plot(results['g'].history['val_acc'], 'g', label='val acc')
#
# loss_ax.set_xlabel('iterlation')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()


from sklearn.metrics import roc_curve

y_pred_keras = g_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1)).ravel()

# Y_test_1 = Y_test[:,1]
# y_pred_keras_1 = y_pred_keras[:,1]
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)

from sklearn.metrics import auc

g_auc_keras = auc(fpr_keras, tpr_keras)

# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(g_auc_keras))
# # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve with Testing Set')
# plt.legend(loc='best')
# plt.show()
#

### Graph Convolution followed by FC layer
g_fc_model = Sequential()
g_fc_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
                         num_neighbors=num_neighbors, activation='relu',
                         input_shape=(X_train.shape[1],1,)))
g_fc_model.add(Dropout(0.2))
g_fc_model.add(Flatten())
g_fc_model.add(Dense(512, activation='relu'))
g_fc_model.add(Dropout(0.2))
g_fc_model.add(Dense(2, activation='softmax'))

g_fc_model.summary()

g_fc_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
                   metrics=['accuracy'])

results['g_fc'] = g_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_fc_error = 1-results['g_fc'].__dict__['history']['val_acc'][-1]

y_pred_keras = g_fc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1)).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
g_fc_auc_keras = auc(fpr_keras, tpr_keras)



# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(results['g_fc'].history['loss'], 'y', label='train loss')
# loss_ax.plot(results['g_fc'].history['val_loss'], 'r', label='val loss')
#
# acc_ax.plot(results['g_fc'].history['acc'], 'b', label='train acc')
# acc_ax.plot(results['g_fc'].history['val_acc'], 'g', label='val acc')
#
# loss_ax.set_xlabel('iterlation')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuray')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()
#
#
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(g_fc_auc_keras))
# # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve with Testing Set')
# plt.legend(loc='best')
# plt.show()
#


# %%

### 2 layers of GCNN with 1 FC

## Graph Convolution followed by FC layer
g_g_fc_model = Sequential()
g_g_fc_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
                         num_neighbors=num_neighbors, activation='relu',
                         input_shape=(X_train.shape[1],1,)))
# g_g_fc_model.add(Dropout(0.2))
# g_g_fc_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
#                         num_neighbors=num_neighbors, activation='relu'))
g_g_fc_model.add(Dropout(0.2))
g_g_fc_model.add(Flatten())
g_g_fc_model.add(Dense(512, activation='relu'))
g_g_fc_model.add(Dropout(0.2))
g_g_fc_model.add(Dense(256, activation='relu'))
g_g_fc_model.add(Dropout(0.2))
g_g_fc_model.add(Dense(2, activation='softmax'))

g_g_fc_model.summary()

g_g_fc_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
                   metrics=['accuracy'])

results['g_fc'] = g_g_fc_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_g_fc_error = 1-results['g_fc'].__dict__['history']['val_acc'][-1]

y_pred_keras = g_g_fc_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1)).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
g_g_fc_auc_keras = auc(fpr_keras, tpr_keras)


### 2 Layer of Graph Convolution
g_g_model = Sequential()
g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
                        num_neighbors=num_neighbors, activation='relu',
                        use_bias=True, input_shape=(X_train.shape[1],1,)))
g_g_model.add(Dropout(0.2))
g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
                        num_neighbors=num_neighbors, activation='relu'))
g_g_model.add(Dropout(0.2))
g_g_model.add(Flatten())
g_g_model.add(Dense(2, activation='softmax'))

g_g_model.summary()

g_g_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
                  metrics=['accuracy'])

results['g_g'] = g_g_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose = 2,
                          validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))

g_g_error = 1-results['g_g'].__dict__['history']['val_acc'][-1]
y_pred_keras = g_g_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1)).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
g_g_auc_keras = auc(fpr_keras, tpr_keras)


fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(results['g_g'].history['loss'], 'y', label='train loss')
loss_ax.plot(results['g_g'].history['val_loss'], 'r', label='val loss')

acc_ax.plot(results['g_g'].history['acc'], 'b', label='train acc')
acc_ax.plot(results['g_g'].history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('iteration')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(g_g_auc_keras))
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve with Testing Set')
plt.legend(loc='best')
plt.show()

np.save("GCNN_fpr50.npy", fpr_keras)
np.save("GCNN_tpr50.npy", tpr_keras)
np.save("GCNN_auc50.npy", g_g_auc_keras)


# %%


### 3 Layers GCNN

### 2 Layer of Graph Convolution
# g_g_g_model = Sequential()
# g_g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
#                         num_neighbors=num_neighbors, activation='relu',
#                         use_bias=True, input_shape=(X_train.shape[1],1,)))
# g_g_g_model.add(Dropout(0.2))
# g_g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
#                         num_neighbors=num_neighbors, activation='relu'))
# g_g_g_model.add(Dropout(0.2))
# g_g_g_model.add(GraphConv(filters=filters, neighbors_ix_mat = graph_mat,
#                         num_neighbors=num_neighbors, activation='relu'))
# g_g_g_model.add(Dropout(0.2))
# g_g_g_model.add(Flatten())
# g_g_g_model.add(Dense(2, activation='softmax'))
#
# g_g_g_model.summary()
#
# g_g_g_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
#                   metrics=['accuracy'])
#
# results['g_g'] = g_g_g_model.fit(X_train.reshape(X_train.shape[0],X_train.shape[1],1), Y_train,
#                           epochs=epochs,
#                           batch_size=batch_size,
#                           verbose = 2,
#                           validation_data=(X_test.reshape(X_test.shape[0],X_test.shape[1],1), Y_test))
#
# g_g_g_error = 1-results['g_g'].__dict__['history']['val_acc'][-1]
# y_pred_keras = g_g_g_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1],1)).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
# g_g_g_auc_keras = auc(fpr_keras, tpr_keras)

# ### Fully Connected - FC Model
# FC_FC_model = Sequential()
# FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# FC_FC_model.add(Dropout(0.2))
# FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# FC_FC_model.add(Dropout(0.2))
# FC_FC_model.add(Dense(2))
# FC_FC_model.add(Activation('softmax'))
#
# FC_FC_model.summary()
#
# FC_FC_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
#                     metrics=['accuracy'])
#
# results['fc_fc'] = FC_FC_model.fit(X_train, Y_train,
#                                   epochs=epochs,
#                                   batch_size=batch_size,
#                                   verbose = 2,
#                                   validation_data=(X_test, Y_test))
#
# fc_fc_error = 1-results['fc_fc'].__dict__['history']['val_acc'][-1]
# y_pred_keras = FC_FC_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1])).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
# fc_fc_auc_keras = auc(fpr_keras, tpr_keras)
# # %%
#
# ### Fully Connected - FC Model
# FC_FC_FC_model = Sequential()
# FC_FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# FC_FC_FC_model.add(Dropout(0.2))
# FC_FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# FC_FC_FC_model.add(Dropout(0.2))
# FC_FC_FC_model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
# FC_FC_FC_model.add(Dropout(0.2))
# FC_FC_FC_model.add(Dense(2))
# FC_FC_FC_model.add(Activation('softmax'))
#
# FC_FC_FC_model.summary()
#
# FC_FC_FC_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
#                     metrics=['accuracy'])
#
# results['fc_fc'] = FC_FC_FC_model.fit(X_train, Y_train,
#                                   epochs=epochs,
#                                   batch_size=batch_size,
#                                   verbose = 2,
#                                   validation_data=(X_test, Y_test))
#
# fc_fc_fc_error = 1-results['fc_fc'].__dict__['history']['val_acc'][-1]
# y_pred_keras = FC_FC_FC_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1])).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
# fc_fc_fc_auc_keras = auc(fpr_keras, tpr_keras)


### Logistic Regression Model
# LR_model = Sequential()
# LR_model.add(Dense(2, activation='relu', input_shape=(X_train.shape[1],)))
# LR_model.add(Activation('softmax'))
#
# LR_model.summary()
#
# LR_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(),
#                  metrics=['accuracy'])
#
# results['lr'] = LR_model.fit(X_train, Y_train,
#                                 epochs=300,
#                                 batch_size=batch_size,
#                                 verbose = 2,
#                                 validation_data=(X_test, Y_test))
#
# lr_error = 1-results['lr'].__dict__['history']['val_acc'][-1]
# y_pred_keras = LR_model.predict(X_test.reshape(X_test.shape[0],X_test.shape[1])).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)
# lr_auc_keras = auc(fpr_keras, tpr_keras)


# %%
print('Error rates for the different models:')
# print('Logistic Regression: %.04f'%lr_error)
print('1 Layer of graph convolution: %.04f'%g_error)
print('2 Layers of graph convolution: %.04f'%g_g_error)
print('1 Layers f graph convolution & 1 FC layer: %.04f'%g_fc_error)
# print('2 FC layers: %.04f'%fc_fc_error)

print('AUC for the different models:')
# print('Logistic Regression: %.04f'%lr_auc_keras)
print('1 Layer of graph convolution: %.04f'%g_auc_keras)
print('2 Layers of graph convolution: %.04f'%g_g_auc_keras)
# print('3 Layers of graph convolution: %.04f'%g_g_g_auc_keras)
print('1 Layers f graph convolution & 1 FC layer: %.04f'%g_fc_auc_keras)
print('1 Layers f graph convolution & 2 FC layer: %.04f'%g_g_fc_auc_keras)
# print('2 FC layers: %.04f'%fc_fc_auc_keras)
# print('3 FC layers: %.04f'%fc_fc_fc_auc_keras)


pickle.dump(results, open('results/MNIST_results.p','wb'))
