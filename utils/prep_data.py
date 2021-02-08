import numpy as np

def train_val_loo_split(preictal_X, preictal_y, interictal_X, interictal_y, val_ratio):
    '''
    Prepare data for leave-one-out cross-validation
    :param preictal_X: List of preprocessed preictal data, split by seizure.
    :param preictal_y: List of preictal labels, split by seizure.
    :param interictal_X: Preprocessed interictal data.
    :param interictal_y: Interictal labels.
    :return: (X_train, y_train, X_val, y_val, X_test, y_test)
    '''
    #For each fold, one seizure is taken out for testing, the rest for training
    #Interictal are concatenated and split into N (no. of seizures) parts,
    #each interictal part is combined with one seizure

    nfold = len(preictal_y)
    rng = np.random.default_rng() # used for shuffling data

    interictal_fold_len = interictal_y.shape[0] // nfold
    print ('Interictal samples in each fold: {}'.format(interictal_fold_len))

    for i in range(nfold):
        if i > 0:
            del X_train
            del y_train
            del X_test
            del y_test
            del X_validation
            del y_validation
            
        X_test_preictal = preictal_X[i]
        y_test_preictal = preictal_y[i]

        X_test_interictal = interictal_X[i*interictal_fold_len:(i+1)*interictal_fold_len]
        y_test_interictal = interictal_y[i*interictal_fold_len:(i+1)*interictal_fold_len]

        if i==0:
            X_train_preictal = np.concatenate(preictal_X[1:])
            y_train_preictal = np.concatenate(preictal_y[1:])

            X_train_interictal = interictal_X[interictal_fold_len:]
            y_train_interictal = interictal_y[interictal_fold_len:]
        elif i < nfold-1:
            X_train_preictal = np.concatenate(preictal_X[:i] + preictal_X[i + 1:], axis=0)
            y_train_preictal = np.concatenate(preictal_y[:i] + preictal_y[i + 1:], axis=0)

            X_train_interictal = np.concatenate([interictal_X[:i * interictal_fold_len], interictal_X[(i + 1) * interictal_fold_len + 1:]],axis=0)
            y_train_interictal = np.concatenate([interictal_y[:i * interictal_fold_len], interictal_y[(i + 1) * interictal_fold_len + 1:]],axis=0)
        else:
            X_train_preictal = np.concatenate(preictal_X[:i], axis=0)
            y_train_preictal = np.concatenate(preictal_y[:i], axis=0)

            X_train_interictal = interictal_X[:i * interictal_fold_len]
            y_train_interictal = interictal_y[:i * interictal_fold_len]

        # remove overlapped ictal samples in test-set
        X_test_preictal = X_test_preictal[y_test_preictal != 2]
        y_test_preictal = y_test_preictal[y_test_preictal != 2]
        X_test_interictal = X_test_interictal[y_test_interictal != 2]
        y_test_interictal = y_test_interictal[y_test_interictal != 2]

        # let overlapped ictal samples have same labels with non-overlapped samples
        y_train_preictal[y_train_preictal == 2] = 1
        y_train_interictal[y_train_interictal == 2] = 1

        print('Preictal size {}, Interictal size {}'.format(y_train_preictal.shape,y_train_interictal.shape))

        '''
        "Downsampling" interictal training set so that the 2 classes
        are balanced
        '''
        down_spl = y_train_interictal.shape[0] // y_train_preictal.shape[0]
        if down_spl > 1:
            X_train_interictal = X_train_interictal[::down_spl]
            y_train_interictal = y_train_interictal[::down_spl]
        elif down_spl == 1:
            X_train_interictal = X_train_interictal[:X_train_preictal.shape[0]]
            y_train_interictal = y_train_interictal[:X_train_preictal.shape[0]]

        print('After balancing: Preictal size {}, Interictal size {}'.format(y_train_preictal.shape,y_train_interictal.shape))

        # shuffle the training data
        idx = np.arange(X_train_preictal.shape[0])
        rng.shuffle(idx)
        X_train_preictal = X_train_preictal[idx]
        y_train_preictal = y_train_preictal[idx]
        rng.shuffle(idx)
        X_train_interictal = X_train_interictal[idx]
        y_train_interictal = y_train_interictal[idx]

        # split training data into training and validation sets
        preictal_split = int(X_train_preictal.shape[0]*(1-val_ratio))
        interictal_split = int(X_train_interictal.shape[0]*(1-val_ratio))

        X_train = np.vstack((X_train_preictal[:preictal_split], X_train_interictal[:interictal_split]))
        y_train = np.concatenate((y_train_preictal[:preictal_split], y_train_interictal[:interictal_split]))

        X_validation = np.vstack((X_train_preictal[preictal_split:], X_train_interictal[interictal_split:]))
        y_validation = np.concatenate((y_train_preictal[preictal_split:], y_train_interictal[interictal_split:]))

        # ensure that # of validation samples is a multiple of 4
        nb_val = X_validation.shape[0] - X_validation.shape[0]%4
        X_validation = X_validation[:nb_val]
        y_validation = y_validation[:nb_val]

        # combine preictal and interictal data for test set
        print('Test samples: Preictal {}, Interictal {}'.format(X_test_preictal.shape[0], X_test_interictal.shape[0]))
        X_test = np.vstack((X_test_interictal, X_test_preictal))
        y_test = np.concatenate((y_test_interictal, y_test_preictal))

        # # shuffle test set
        # idx = np.arange(X_test.shape[0])
        # rng.shuffle(idx)
        # X_test = X_test[idx]
        # y_test = y_test[idx]

        print ('Shapes: X_train {}, X_val {}, X_test {}'.format(X_train.shape, X_validation.shape, X_test.shape))

        # adding a dimension because old code is a 3D CNN
        # delete below when CNN is converted to 2D
        # X_train = np.expand_dims(X_train, axis=1)
        # X_validation = np.expand_dims(X_validation, axis=1)
        # X_test = np.expand_dims(X_test, axis=1)

        yield (X_train, y_train, X_validation, y_validation, X_test, y_test)


def train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio, test_ratio):

    num_sz = len(ictal_y)
    num_sz_test = int(test_ratio*num_sz)
    print ('Total %d seizures. Last %d is used for testing.' %(num_sz, num_sz_test))

    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X,axis=0)
        interictal_y = np.concatenate(interictal_y,axis=0)
    interictal_fold_len = int(round(1.0*interictal_y.shape[0]/num_sz))
    print ('interictal_fold_len',interictal_fold_len)


    X_test_ictal = np.concatenate(ictal_X[-num_sz_test:])
    y_test_ictal = np.concatenate(ictal_y[-num_sz_test:])

    X_test_interictal = interictal_X[-num_sz_test*interictal_fold_len:]
    y_test_interictal = interictal_y[-num_sz_test*interictal_fold_len:]

    X_train_ictal = np.concatenate(ictal_X[:-num_sz_test],axis=0)
    y_train_ictal = np.concatenate(ictal_y[:-num_sz_test],axis=0)

    X_train_interictal = interictal_X[:-num_sz_test*interictal_fold_len]
    y_train_interictal = interictal_y[:-num_sz_test*interictal_fold_len]

    print (y_train_ictal.shape,y_train_interictal.shape)

    '''
    Downsampling interictal training set so that the 2 classes
    are balanced
    '''
    down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
    if down_spl > 1:
        X_train_interictal = X_train_interictal[::down_spl]
        y_train_interictal = y_train_interictal[::down_spl]
    elif down_spl == 1:
        X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
        y_train_interictal = y_train_interictal[:X_train_ictal.shape[0]]

    print ('Balancing:', y_train_ictal.shape,y_train_interictal.shape)

    #X_train_ictal = shuffle(X_train_ictal,random_state=0)
    #X_train_interictal = shuffle(X_train_interictal,random_state=0)
    X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))],X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)
    y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]),axis=0)

    X_val = np.concatenate((X_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):],X_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)
    y_val = np.concatenate((y_train_ictal[int(X_train_ictal.shape[0]*(1-val_ratio)):], y_train_interictal[int(X_train_interictal.shape[0]*(1-val_ratio)):]),axis=0)

    nb_val = X_val.shape[0] - X_val.shape[0]%4
    X_val = X_val[:nb_val]
    y_val = y_val[:nb_val]

    # let overlapped ictal samples have same labels with non-overlapped samples
    y_train[y_train==2] = 1
    y_val[y_val==2] = 1

    X_test = np.concatenate((X_test_ictal, X_test_interictal),axis=0)
    y_test = np.concatenate((y_test_ictal, y_test_interictal),axis=0)

    # remove overlapped ictal samples in test-set
    X_test = X_test[y_test != 2]
    y_test = y_test[y_test != 2]

    print ('X_train, X_val, X_test',X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test
