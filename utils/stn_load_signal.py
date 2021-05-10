import os, csv
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample, stft

import matplotlib.pyplot as plt
from matplotlib import cm

import hickle as hkl

class PrepData():
    def __init__(self, target, type, settings, clipTarget=None, clipLength=30):
        self.target = target  # patient/subject name e.g. 'Patient_1'
        self.settings = settings  # settings from json.load
        self.phase = type  # preictal or interictal
        self.phaseFiles = []  # list of eeg files for this phase
        self.clipTarget = clipTarget  # desired number of clips per file (only used to oversample)
        self.clipLength = clipLength  # length of each clip in seconds

    def load_signals_Kaggle2014Pred(self, data_dir, target, data_type):
        '''
        Generator for loading EEG data from Kaggle dataset.
        Files should be in the format 'target'_'data_type'_segment_####.mat
            e.g. Patient_1_preictal_segment_0012.mat

        Parameters:
            data_dir:   path to directory for the target files
            target:     name of subject e.g. 'Patient_1'
            data_type:  type of files to be loaded, e.g. 'preictal'
        '''

        reader = csv.reader(open('kaggle_Segment_Sequence.csv', mode='r'))
        d = {}
        for file, sequence in reader:
            d[file] = sequence

        dir = os.path.join(data_dir, target)
        alarm_start = 3  # start of alarm period, in terms of (10n + 5) minutes before seizure onset
        done = False
        i = 0

        # load each file for the data_type e.g. all preictal segments for Patient_1
        while not done:
            # create string for path/filename.
            i += 1
            nstr = '{:04}'.format(i)  # number must be 4 digits; left padded with 0 otherwise
            filename = '{}_{}_segment_{}.mat'.format(target, data_type, nstr)
            filepath = '{}/{}'.format(dir, filename)

            # data is in format [EEG_data, data_length_sec, sampling_freq, channels, sequence]
            if os.path.exists(filepath):
                data = scipy.io.loadmat(filepath)
                # find the actual data in loaded .mat dict, key similar to 'preictal_segment_1'
                d_key = [key for key in data.keys() if '_segment_' in key][0]

                # discard preictal segments from 65 to 35 min prior to seizure
                if data_type == 'preictal':
                    # segment is the (7 - n)th 10 minute segment preceding the 5 min horizon before seizure onset
                    # for example, segment = 1 indicates this data segment covers 65 - 55 min before seizure onset
                    segment = data[d_key][0][0][4][0][0]
                    if (segment <= alarm_start):
                        print('Skipping {}'.format(filepath))
                        continue

                self.phaseFiles.append([filepath, d['{}/{}'.format(target, filename)]])

                yield {'eeg': data[d_key][0][0][0],
                       'slength': data[d_key][0][0][1][0][0],
                       'sfreq': data[d_key][0][0][2][0][0],
                       'channels': data[d_key][0][0][3],
                       'segment': data[d_key][0][0][4][0][0],
                       'sequence': self.phaseFiles[-1][1]
                       }

            # file not found, or looped through all applicable files
            else:
                if i == 1:
                    raise Exception("file %s not found" % filepath)
                done = True


    def load_signals_CHBMIT(self, data_dir, target, data_type):
        print('load_signals_CHBMIT for Patient', target)
        from mne.io import RawArray, read_raw_edf
        # from mne.channels import read_montage
        # from mne import create_info, concatenate_raws, pick_types
        # from mne.filter import notch_filter

        szr_summary = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'), header=0)
        all_preictal_files = list(szr_summary['File_name'])
        szr_start = szr_summary['Seizure_start']
        szr_stop = szr_summary['Seizure_stop']

        segment = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'), header=None)
        interictal_files = list(segment[segment[1] == 0][0])

        nsdict = {
            '0': []
        }
        targets = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',
            '22',
            '23'
        ]
        for t in targets:
            nslist = [f for f in interictal_files if 'chb{:02}'.format(int(t)) in f]
            nsdict[t] = nslist
        # nsfilenames = shuffle(nsfilenames, random_state=0)

        special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'), header=None)
        sifilenames, sistart, sistop = special_interictal[0], special_interictal[1], special_interictal[2]
        sifilenames = list(sifilenames)

        target_folder = 'chb{:02}'.format(int(target))
        dir = os.path.join(data_dir, target_folder)
        edf_files = [f for f in os.listdir(dir) if f.endswith('.edf')]
        # print (target,strtrg)
        print(edf_files)

        if data_type == 'preictal':
            filenames = [f for f in edf_files if f in all_preictal_files]
            # print ('preictal files', filenames)
        elif data_type == 'interictal':
            filenames = [f for f in edf_files if f in nsdict[target]]
            # print ('interictal files', filenames)

        totalfiles = len(filenames)
        print('Total {} files {}'.format(data_type, totalfiles))

        for fname in filenames:
            exclude_chs = []
            if target in ['4', '9']:
                exclude_chs = [u'T8-P8']

            if target in ['13', '16']:
                chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4',
                       u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
            elif target in ['4']:
                chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4',
                       u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7',
                       u'T7-FT9', u'FT10-T8']
            else:
                chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4',
                       u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ',
                       u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']

            rawEEG = read_raw_edf('{}/{}'.format(dir, fname),
                                  exclude=exclude_chs,
                                  verbose=0, preload=True)

            rawEEG.pick_channels(chs)
            tmp = rawEEG.to_data_frame()
            tmp = tmp.to_numpy()

            if data_type == 'preictal':
                SOP = 30 * 60 * 256
                # get seizure szr_summary information
                indices = [ind for ind, x in enumerate(all_preictal_files) if x == fname]
                if len(indices) > 0:
                    print('{} seizures in the file {}'.format(len(indices), fname))
                    prev_sp = -1e6
                    for i in range(len(indices)):
                        st = szr_start[indices[i]] * 256 - 5 * 60 * 256  # SPH=5min
                        sp = szr_stop[indices[i]] * 256
                        print('Seizure %s %d starts at %d stops at %d last sz stop is %d' % (fname, i, (st+5*60*256),sp,prev_sp))

                        # take care of some special filenames
                        if fname[6] == '_':
                            seq = int(fname[7:9])
                        else:
                            seq = int(fname[6:8])
                        if fname == 'chb02_16+.edf':
                            prevfile = 'chb02_16.edf'
                        else:
                            if fname[6] == '_':
                                prevfile = '{}_{:02}.edf'.format(fname[:6], seq - 1)
                            else:
                                prevfile = '{}_{:02}.edf'.format(fname[:5], seq - 1)

                        if st - SOP > prev_sp:
                            prev_sp = sp
                            if st - SOP >= 0:
                                data = tmp[st - SOP: st]
                            else:
                                if os.path.exists('{}/{}'.format(dir, prevfile)):
                                    rawEEG = read_raw_edf('{}/{}'.format(dir, prevfile), preload=True, verbose=0)
                                    rawEEG.pick_channels(chs)
                                    prevtmp = rawEEG.to_data_frame()
                                    prevtmp = prevtmp.to_numpy()
                                    if st > 0:
                                        data = np.concatenate((prevtmp[st - SOP:], tmp[:st]))
                                    else:
                                        data = prevtmp[st - SOP:st]

                                else:
                                    if st > 0:
                                        data = tmp[:st]
                                    else:
                                        # raise Exception("file %s does not contain useful info" % filename)
                                        print("WARNING: file {} does not contain useful info".format(fname))
                                        continue
                        else:
                            prev_sp = sp
                            continue

                        print('data shape', data.shape)
                        if data.shape[0] == SOP:
                            # yield (data)
                            return (data)
                        else:
                            continue

            elif data_type == 'interictal':
                if fname in sifilenames:
                    st = sistart[sifilenames.index(fname)]
                    sp = sistop[sifilenames.index(fname)]
                    if sp < 0:
                        data = tmp[st * 256:]
                    else:
                        data = tmp[st * 256:sp * 256]
                else:
                    data = tmp
                print('data shape', data.shape)
                # yield (data)
                return (data)

    def read_raw_signal(self):
        '''Load EEG data from raw files. Calls separate helper functions for each dataset.'''

        if self.settings['dataset'] == 'CHBMIT':
            self.samp_freq = 256
            self.freq = 256
            self.global_proj = np.array([0.0]*114)
            self.load_signals_CHBMIT(self.settings['datadir'], self.target, self.phase)

        if self.settings['dataset'] == 'Kaggle2014Pred':
            return self.load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, self.phase)

        return 'array, freq, misc'

    def _eegGenerator(self):
        """
        Generator for loading EEG data files. Assumes self.phaseFiles is not empty
        """
        for filename, sequence in self.phaseFiles:
            if self.settings['dataset'] == 'Kaggle2014Pred':
                data = scipy.io.loadmat(filename)
                # find the actual data in loaded .mat dict, key similar to 'preictal_segment_1'
                d_key = [key for key in data.keys() if '_segment_' in key][0]

            elif self.settings['dataset'] == 'CHBMIT':
                print()

            yield {'eeg': data[d_key][0][0][0],
                   'slength': data[d_key][0][0][1][0][0],
                   'sfreq': data[d_key][0][0][2][0][0],
                   'channels': data[d_key][0][0][3],
                   'segment': data[d_key][0][0][4][0][0],
                   'sequence': sequence
                   }

    def oversample(self, extraClips):
        """
        Oversample EEG using sliding window.
        """

        perFile = extraClips // len(self.phaseFiles)
        data = self._eegGenerator()

        print('Generating {} extra clips...'.format(perFile * len(self.phaseFiles)))

        X, y = self.splitClips(data, perFile)
        return X, y

    def splitClips(self, dataset, clipTarget=None):
        """
        Input should be a generator for dicts with the form (eeg_data, sample_length, sample_freq, channels, segment)
        """
        alreadyPrinted = False
        print('Preprocessing data for {} {}'.format(self.phase, self.target))

        if self.settings['dataset'] == 'Kaggle2014Pred':
            if 'Dog_' in self.target:
                targetFrequency = 200
            else:
                targetFrequency = 1000

        elif self.settings['dataset'] == 'CHBMIT':
            targetFrequency = 256

        window = self.clipLength * targetFrequency

        xVals = {}
        y = {}

        for sample in dataset:
            # downsample to target frequency
            if sample['sfreq'] != targetFrequency:
                eeg_data = []
                for channel in sample['eeg']:
                    eeg_data.append(resample(channel, targetFrequency * sample['slength'], axis=-1))
                eeg_data = np.asarray(eeg_data)
            else:
                eeg_data = np.asarray(sample['eeg'])

            # used for oversampling
            if clipTarget is not None:
                stride = (eeg_data.shape[-1] - window) // (clipTarget - 1)
                if not alreadyPrinted:
                    print('Oversampling stride is {} samples (signal is sampled at {} Hz)'.format(stride, targetFrequency))
                    alreadyPrinted = True
                clipTotal = clipTarget - 1

            else:
                stride = window
                clipTotal = eeg_data.shape[-1] // window

            # split into smaller clips
            x = []
            for i in range(clipTotal):
                if clipTarget is not None:
                    clip = eeg_data[:, ((i+1) * stride) : ((i+1) * stride + window)]
                else:
                    clip = eeg_data[:, (i*stride):(i*stride + window)]

                # # plot raw eeg data
                # plt.plot(np.arange(0,30,1/targetFrequency), clip[0,:])
                # plt.title('Raw EEG (clip {})'.format(i))
                # plt.ylabel('Magnitude')
                # plt.xlabel('Time [sec]')
                # plt.show

                x.append(clip)

            seq = sample['sequence']
            xVals.setdefault(seq, []).extend(np.asarray(x))

            # assign class for processed clips
            if clipTarget is not None:
                y.setdefault(seq, []).extend([2] * clipTotal)
            elif self.phase == 'preictal':
                y.setdefault(seq, []).extend([1]*clipTotal)
            else:
                y.setdefault(seq, []).extend([0]*clipTotal)

        # convert to numpy array
        for sequence in xVals:
            xVals[sequence] = np.asarray(xVals[sequence])
            y[sequence] = np.asarray(y[sequence])

        return xVals, y

    def cacheData(self, X, y):
        filename = 'stn_{self.phase}_{self.target}.hkl'.format(self=self)  # e.g. ictal_Patient_1.hkl
        filepath = os.path.join(self.settings['cachedir'], filename)
        print('Caching {}...'.format(filename))
        hkl.dump([X, y], filepath, mode='w')
        print('Done caching')

    def apply(self):
        filename = 'stn_{self.phase}_{self.target}.hkl'.format(self=self)  # e.g. ictal_Patient_1.hkl
        filepath = os.path.join(self.settings['cachedir'], filename)

        # load clips from cache (if it exists)
        if os.path.isfile(filepath):
            print('Loading {} from cache...'.format(filename))
            X, y = hkl.load(filepath)
            print('Done loading')

        # split data into clips
        else:
            data = self.read_raw_signal()
            X, y = self.splitClips(data)

            print('Caching {}...'.format(filename))
            hkl.dump([X, y], filepath, mode='w')
            print('Done caching')

        return X, y
