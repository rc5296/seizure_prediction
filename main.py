import json
import os
import os.path
import numpy as np
import pandas as pd
# import keras
import tensorflow.keras as keras
keras.backend.set_image_data_format('channels_first')
print ('Using Keras image_data_format={}'.format(keras.backend.image_data_format()))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from utils.load_signals import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split
from models.cnn import ConvNN

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def main(dataset='Kaggle2014Pred', build_type='cv'):
    print ('Main')
    with open('SETTINGS_%s.json' %dataset) as f:
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
            'Patient_1'
            # 'Patient_2'
        ]

    for target in targets:
        # create data structures for preictal and interictal data
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
            print('Generating {} extra clips...'.format(extraClips))

            temp_X, temp_y = preictal.oversample(extraClips)

            for i in range(len(preictal_X)):
                for sequence in preictal_X[i]:
                    temp_X[i][sequence] = np.concatenate((preictal_X[i][sequence], temp_X[i][sequence]))

            preictal_X = temp_X
            del temp_X

            for sequence in preictal_y:
                preictal_y[sequence] = np.concatenate((preictal_y[sequence], temp_y[sequence]))
            del temp_y

            preictal.cacheData(preictal_X, preictal_y)

        # change preictal data to list
        for i in range(len(preictal_X)):
            preictal_X[i] = [v for _, v in preictal_X[i].items()]
        preictal_y = [v for _, v in preictal_y.items()]

        # concat interictal data
        for i in range(len(interictal_X)):
            interictal_X[i] = [q for _, v in interictal_X[i].items() for q in v]
            interictal_X[i] = np.asarray(interictal_X[i])
        interictal_y = [q for _, v in interictal_y.items() for q in v]
        interictal_y = np.asarray(interictal_y)
        # temp_X = [[],[],[]]
        # for i, d in enumerate(interictal_X):
        #     for j in d.values():
        #         temp_X[i].extend(j)
        #     temp_X[i] = np.asarray(temp_X[i])
        # interictal_X = temp_X
        # del temp_X
        #
        # temp_y = []
        # for i in interictal_y.values():
        #     temp_y.extend(i)
        # interictal_y = np.asarray(temp_y)
        # del temp_y

        # train model from scratch
        if build_type=='cv':
            auc_folds = []
            tpos_folds = []
            fpos_folds = []

            loo_folds = train_val_loo_split(preictal_X[-1], preictal_y, interictal_X[-1], interictal_y, 0.25)
            ind = 1
            for X_train, y_train, X_val, y_val, X_test, y_test in loo_folds:
                print('X_train, y_train, X_val, y_val, X_test, y_test')
                print (X_train.shape, y_train.shape,
                       X_val.shape, y_val.shape,
                       X_test.shape, y_test.shape)

                model = ConvNN(target,batch_size=16,nb_classes=2,epochs=50,mode=build_type)
                model.setup(X_train.shape)
                model.fit(X_train, y_train, X_val, y_val)
                auc, tpos, fpos, predictions = model.evaluate(X_test, y_test)

                auc_folds.append(auc)
                tpos_folds.append(tpos)
                fpos_folds.append(fpos)

                # write out predictions for preictal and interictal segments
                results_df = pd.DataFrame(np.hstack((y_test[:,None], predictions[:,None])))
                filename = os.path.join(str(settings['resultdir']), '{}_{}.xlsx'.format(target, ind))

                results_df.to_excel(filename, header=['ground truth','prediction'], index=False)
                print('wrote results to {}'.format(filename))
                '''
                # preictal
                X_test_p = X_test[y_test==1]
                y_test_p = model.predict_proba(X_test_p)
                filename = os.path.join(
                    str(settings['resultdir']), 'preictal_%s_%d.csv' %(target, ind))
                lines = []
                lines.append('preictal')
                for i in range(len(y_test_p)):
                    lines.append('%.4f' % ((y_test_p[i][1])))
                with open(filename, 'w') as f:
                    f.write('\n'.join(lines))
                print('wrote results to {}'.format(filename))

                # interictal
                X_test_i = X_test[y_test==0]
                y_test_i = model.predict_proba(X_test_i)
                filename = os.path.join(
                    str(settings['resultdir']), 'interictal_%s_%d.csv' %(target, ind))
                lines = []
                lines.append('interictal')
                for i in range(len(y_test_i)):
                    lines.append('%.4f' % ((y_test_i[i][1])))
                with open(filename, 'w') as f:
                    f.write('\n'.join(lines))
                print('wrote results to {}'.format(filename))
                '''

                ind += 1

            for i in range(len(auc_folds)):
                print('===============================================')
                print('Fold {}:'.format(i+1))
                print('AUC={}\nTrue Pos={}\nFalse Pos={}'.format(auc_folds[i],tpos_folds[i],fpos_folds[i]))


            print('===============================================')
            print('Average AUC={}'.format(np.mean(auc_folds)))
            print('Total Sensitivity={}'.format(sum(tpos_folds) / len(tpos_folds)))
            print('Total false positives={}'.format(sum(fpos_folds)))


        # elif build_type=='test':
        #     X_train, y_train, X_val, y_val, X_test, y_test = \
        #         train_val_test_split(preictal_X, preictal_y, interictal_X, interictal_y, 0.25, 0.35)
        #     model = ConvNN(target,batch_size=32,nb_classes=2,epochs=100,mode=build_type)
        #     model.setup(X_train.shape)
        #     #model.fit(X_train, y_train)
        #     fn_weights = "weights_%s_%s.h5" %(target, build_type)
        #     if os.path.exists(fn_weights):
        #         model.load_trained_weights(fn_weights)
        #     else:
        #         model.fit(X_train, y_train, X_val, y_val)
        #     model.evaluate(X_test, y_test)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", help="cv or test. cv is for leave-one-out cross-validation")
#     parser.add_argument("--dataset", help="FB, CHBMIT or Kaggle2014Pred")
#     args = parser.parse_args()
#     assert args.mode in ['cv','test']
#     main(dataset=args.dataset, build_type=args.mode)

main()