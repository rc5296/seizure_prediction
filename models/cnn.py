import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from models.stn import SequenceTransformer, STFT

class ConvNN(object):
    def __init__(self, target, batch_size=16, nb_classes=2, epochs=2, mode='base'):
        self.target = target
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.epochs = epochs
        self.mode = mode

    def setup(self, X_train_shape):
        print('X_train shape', X_train_shape)

        inputs = keras.layers.Input(shape=X_train_shape[1:])

        if self.mode == 'stn':
            x, theta_phi = SequenceTransformer('stn')(inputs)
            layer = STFT()(x)
        else:
            layer = inputs

        # Tensors should have dimension (N, C, F, T) at this point
        # C = channels, F = frequency, T = time
        normal1 = BatchNormalization(axis=1, name='normal1')(layer)
        conv1 = Convolution2D(16, (5, 5), padding='valid', strides=(2, 2), name='conv1')(normal1)
        relu1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(relu1)

        normal2 = BatchNormalization(axis=1, name='normal2')(pool1)
        conv2 = Convolution2D(32, (3, 3), padding='same', strides=(1, 1), name='conv2')(normal2)
        relu2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(relu2)

        normal3 = BatchNormalization(axis=1, name='normal3')(pool2)
        conv3 = Convolution2D(64, (3, 3), padding='same', strides=(1, 1), name='conv3')(normal3)
        relu3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(relu3)

        flat = Flatten()(pool3)

        drop1 = Dropout(0.5)(flat)
        dens1 = Dense(128, activation='sigmoid', name='dens1')(drop1)
        drop2 = Dropout(0.5)(dens1)
        dens2 = Dense(self.nb_classes, name='dens2')(drop2)

        last = Activation('softmax')(dens2)

        self.model = keras.models.Model(inputs=inputs, outputs=last)
        adam = keras.optimizers.Adam('clipnorm', lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy', keras.metrics.AUC()])

        print(self.model.summary())
        return self

    def fit(self, X_train, Y_train, X_val, y_val, save_model=False, load_model=False):
        Y_train = Y_train.astype('uint8')
        Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
        y_val = np_utils.to_categorical(y_val, self.nb_classes)

        # callbacks
        cb = []
        # Create a callback for early stopping based on validation loss
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
        cb.append(early_stop)

        if save_model:
            # os.remove('{}_{}.ckpt'.format(self.target, self.mode))
            # Create a callback that saves the model's weights
            ckpt_callback = keras.callbacks.ModelCheckpoint(filepath='{}_{}.ckpt'.format(self.target, self.mode),
                                                            save_weights_only=True,
                                                            monitor='val_loss',
                                                            mode='min',
                                                            save_best_only=True)
            cb.append(ckpt_callback)

        self.model.fit(X_train,
                       Y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(X_val, y_val),
                       callbacks=cb,
                       verbose=2)

        if load_model:
            self.model.load_weights('{}_{}.ckpt'.format(self.target, self.mode))

        return self

    def predict_proba(self, X):
        return self.model.predict([X])

    def evaluate(self, X, y, k_out_of_n=(6, 8), plot_roc=False):
        predictions = self.model.predict(X, verbose=0)

        auc_test = metrics.roc_auc_score(y, predictions[:, 1])
        print('Test AUC is:{}'.format(auc_test))

        # Plot ROC curve
        if plot_roc:
            fpr, tpr, threshold = metrics.roc_curve(y, predictions[:,1])
            roc_auc = metrics.auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic (Patient 1, Seizure   )')
            plt.plot(fpr, tpr, 'b', label='AUC = {:.2f}'.format(roc_auc))
            plt.legend(loc='lower right')
            plt.plot([0,1], [0,1], 'r--')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

        class_predict = np.argmax(predictions, axis=1)
        pos_idx = np.nonzero(y == 1)
        neg_idx = np.nonzero(y == 0)

        true_pos = getTruePos(class_predict[pos_idx], k_out_of_n)
        false_pos = getFalsePos(class_predict[neg_idx], k_out_of_n)

        return auc_test, true_pos, false_pos, class_predict


def getTruePos(predictions, k_out_of_n):
    """
    Check if the model correctly predicted a seizure. Model must make "k-out-of-n" correct predictions on
    the preictal data clips. Assumes that leave-one-out cross validation is used i.e. only preictal data
    of one seizure is input.

    :param predictions: Predictions made by the model for preictal data clips of one seizure
    :param k_out_of_n: Tuple (k,n) for "k-out-of-n" method of determining alarm
    :return: Returns 1 if the model correctly predicted a seizure. Returns 0 otherwise
    """
    count = 0
    j = 0
    nArr = []

    for i in range(len(predictions)):
        if count < k_out_of_n[1]:
            nArr.append(predictions[i])
            count += 1
            if sum(nArr) >= k_out_of_n[0]:  # sound alarm
                return 1

        else:
            nArr[j] = predictions[i]
            if sum(nArr) >= k_out_of_n[0]:  # sound alarm
                return 1
            j = (j + 1) % k_out_of_n[1]

    return 0


def getFalsePos(predictions, k_out_of_n):
    """
    Count how many false alarms the model made. Note: interictal
    clips may not be in sequential order, so "k-out-of-n" is an approximation in this case.

    :param predictions: Predictions made by the model for interictal data clips
    :param k_out_of_n: Tuple (k,n) for "k-out-of-n" method of determining alarm
    :return: Returns the number of false alarms
    """
    alarm = False
    disabledClips = 69  # amount of time the alarm lasts for, where time=(disabledCLips + 1)*clipLength
    disableCount = 0
    fpos = 0
    count = 0
    j = 0
    nArr = []

    for i in range(len(predictions)):
        if not alarm:
            if count < k_out_of_n[1]:
                nArr.append(predictions[i])
                count += 1
                if sum(nArr) >= k_out_of_n[0]:  # sound alarm
                    fpos += 1
                    alarm = True

            else:
                nArr[j] = predictions[i]
                if sum(nArr) >= k_out_of_n[0]:  # sound alarm
                    fpos += 1
                    alarm = True

                j = (j + 1) % k_out_of_n[1]

        # re-enable alarm after alarm period
        elif disableCount == disabledClips:
            disableCount = 0
            count = 0
            nArr = []
            j = 0
            alarm = False

        else:
            disableCount += 1

    return fpos
