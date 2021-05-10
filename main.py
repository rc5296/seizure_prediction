import json, os, sys, argparse
import os.path
import numpy as np
import pandas as pd
# import keras
import tensorflow.keras as keras
keras.backend.set_image_data_format('channels_first')
print ('Using Keras image_data_format={}'.format(keras.backend.image_data_format()))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from utils.prep_data import train_val_loo_split
from models.cnn import ConvNN

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def main(dataset='Kaggle2014Pred', mode='base'):
    if mode =='stn':
        from utils.stn_load_signal import PrepData
    else:
        from utils.load_signals import PrepData

    with open('SETTINGS_{}.json'.format(dataset)) as f:
        settings = json.load(f)

    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset']=='Kaggle2014Pred':
        targets = [
            # 'Dog_1',
            # 'Dog_2',
            # 'Dog_3',
            # 'Dog_4',
            # 'Dog_5'
            'Patient_1',
            # 'Patient_2'
        ]

    elif settings['dataset']=='CHBMIT':
        targets = [
            '1',
            '2'
            # '3',
            # '5',
            # '9',
            # '10',
            # '13',
            # '14',
            # '18',
            # '19',
            # '20',
            # '21',
            # '23'
        ]

    for target in targets:
        # split preictal and interictal data into smaller clips
        preictal = PrepData(target, type='preictal', settings=settings)
        preictal_X, preictal_y = preictal.apply()
        interictal = PrepData(target, type='interictal', settings=settings)
        interictal_X, interictal_y = interictal.apply()

        # calculate total number of examples in each class
        totalPre = 0
        totalInt = 0
        for _, y in preictal_y.items():
            totalPre += y.shape[0]
        for _, y in interictal_y.items():
            totalInt += y.shape[0]

        # generate more preictal data so classes are more balanced
        if (totalInt - totalPre) >= 1e2:
            extraClips = totalInt - totalPre
            temp_X, temp_y = preictal.oversample(extraClips)

            for sequence in preictal_X:
                preictal_X[sequence] = np.concatenate((preictal_X[sequence], temp_X[sequence]))
                preictal_y[sequence] = np.concatenate((preictal_y[sequence], temp_y[sequence]))
            del temp_X
            del temp_y

            preictal.cacheData(preictal_X, preictal_y)

        # change preictal data to list
        preictal_X = [v for _, v in preictal_X.items()]
        preictal_y = [v for _, v in preictal_y.items()]

        # concat interictal data
        interictal_X = [q for _, v in interictal_X.items() for q in v]
        interictal_X = np.asarray(interictal_X)
        interictal_y = [q for _, v in interictal_y.items() for q in v]
        interictal_y = np.asarray(interictal_y)

        # # train model from scratch
        auc_folds = []
        tpos_folds = []
        fpos_folds = []

        loo_folds = train_val_loo_split(preictal_X, preictal_y, interictal_X, interictal_y, 0.25)
        ind = 1

        for X_train, y_train, X_val, y_val, X_test, y_test in loo_folds:
            print('X_train, y_train, X_val, y_val, X_test, y_test')
            print(X_train.shape, y_train.shape,
                  X_val.shape, y_val.shape,
                  X_test.shape, y_test.shape)

            model = ConvNN(target, batch_size=16, nb_classes=2, epochs=50, mode=mode)
            model.setup(X_train.shape)
            model.fit(X_train, y_train, X_val, y_val)
            auc, tpos, fpos, predictions = model.evaluate(X_test, y_test)

            auc_folds.append(auc)
            tpos_folds.append(tpos)
            fpos_folds.append(fpos)

            # write out predictions for preictal and interictal segments
            results_df = pd.DataFrame(np.hstack((y_test[:, None], predictions[:, None])))
            filename = os.path.join(str(settings['resultdir']), '{}_{}.xlsx'.format(target, ind))

            results_df.to_excel(filename, header=['ground truth', 'prediction'], index=False)
            print('wrote results to {}'.format(filename))

            ind += 1

        for i in range(len(auc_folds)):
            print('===============================================')
            print('Fold {}:'.format(i + 1))
            print('AUC={}\nTrue Pos={}\nFalse Pos={}'.format(auc_folds[i], tpos_folds[i], fpos_folds[i]))

        print('===============================================')
        print('Average AUC={}'.format(np.mean(auc_folds)))
        print('Total Sensitivity={}'.format(sum(tpos_folds) / len(tpos_folds)))
        print('Total false positives={}'.format(sum(fpos_folds)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', help='stn or base')
    parser.add_argument('--dataset', help='Kaggle2014Pred')
    args = parser.parse_args()
    main(dataset=args.dataset, mode=args.mode)

# For debugging: comment "if" block above, uncomment below
# main()