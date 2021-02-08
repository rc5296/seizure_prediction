import os, csv
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample, stft

import matplotlib.pyplot as plt
from matplotlib import cm

import hickle as hkl
from utils.group_seizure_Kaggle2014Pred import group_seizure

class PrepData():
    def __init__(self, target, type, settings, clipTarget=None):
        self.target = target                    # patient/subject name e.g. 'Patient_1'
        self.settings = settings                # settings from json.load
        self.phase = type                       # preictal or interictal
        self.phaseFiles = []                    # list of eeg files for this phase
        self.clipTarget = clipTarget            # desired number of clips per file (only used to oversample)

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
                # return ({'eeg': data[d_key][0][0][0],
                #          'slength': data[d_key][0][0][1][0][0],
                #          'sfreq': data[d_key][0][0][2][0][0],
                #          'channels': data[d_key][0][0][3],
                #          'segment': data[d_key][0][0][4][0][0]
                #          },)

            # file not found, or looped through all applicable files
            else:
                if i == 1:
                    raise Exception("file %s not found" % filepath)
                done = True

    def read_raw_signal(self):
        '''Load EEG data from raw files. Calls separate helper functions for each dataset.'''

        # if self.settings['dataset'] == 'CHBMIT':
        #     self.samp_freq = 256
        #     self.freq = 256
        #     self.global_proj = np.array([0.0]*114)
        #     return load_signals_CHBMIT(self.settings['datadir'], self.target, self.type)

        # elif self.settings['dataset'] == 'FB':
        #     self.samp_freq = 256
        #     self.freq = 256
        #     self.global_proj = np.array([0.0]*114)
        #     return load_signals_FB(self.settings['datadir'], self.target, self.type)

        if self.settings['dataset'] == 'Kaggle2014Pred':

            return self.load_signals_Kaggle2014Pred(self.settings['datadir'], self.target, self.phase)

        return 'array, freq, misc'

    def _eegGenerator(self):
        """
        Generator for loading EEG data files. Assumes self.phaseFiles is not empty
        """
        for filename, sequence in self.phaseFiles:
            data = scipy.io.loadmat(filename)
            # find the actual data in loaded .mat dict, key similar to 'preictal_segment_1'
            d_key = [key for key in data.keys() if '_segment_' in key][0]

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

        X, y = self.preprocess(data, perFile)
        return X, y


    def preprocess(self, dataset, clipTarget=None):
        """
        Input should be a generator for dicts with the form (eeg_data, sample_length, sample_freq, channels, segment)
        """
        alreadyPrinted = False
        print('Preprocessing data for {} {}'.format(self.phase, self.target))

        if 'Dog_' in self.target:
            targetFrequency = 200
        else:
            targetFrequency = 1000

        clipLength = 30  # in seconds
        window = clipLength * targetFrequency

        xVals = {}
        freqs = {}
        times = {}
        y = {}

        for sample in dataset:
            # downsample to target frequency
            eeg_data = []
            for channel in sample['eeg']:
                eeg_data.append(resample(channel, targetFrequency*sample['slength'], axis=-1))
            eeg_data = np.asarray(eeg_data)

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

            # stft on smaller clips
            f = []
            t = []
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
                # plt.show()

                fclip, tclip, xclip = self._stft_process(clip, targetFrequency, clip=i)
                f.append(fclip)
                t.append(tclip)
                x.append(xclip)

            seq = sample['sequence']
            freqs.setdefault(seq, []).extend(np.asarray(f))
            times.setdefault(seq, []).extend(np.asarray(t))
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
            freqs[sequence] = np.asarray(freqs[sequence])
            times[sequence] = np.asarray(times[sequence])
            xVals[sequence] = np.asarray(xVals[sequence])
            y[sequence] = np.asarray(y[sequence])

        return [freqs, times, xVals], y

    def _stft_process(self, eegData, sampFreq, clip=0):
        powerFreq = 60
        f, t, Zxx = stft(eegData, sampFreq, window='hann', nperseg=512, axis=-1)

        # plot before
        temp = np.log(np.abs(Zxx[0, :, :]))+1e-6
        plt.pcolormesh(t, f, temp, shading='gouraud', cmap='inferno', vmax=9, vmin=-4.5) # use this for human
        # plt.pcolormesh(t, f, np.log(np.abs(Zxx[0, :, :])) + 1e-6, shading='gouraud', cmap='inferno')
        # plt.title('STFT Magnitude before removal (clip {})'.format(clip))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()
        plt.show()

        # remove DC (0 Hz) and power line harmonics (+/- 3 Hz)
        Zxx = Zxx[:, 1:, :]
        line_noise_freq = []
        # nBins = len(f)
        f = f[1:]
        for i in range(int(np.max(f) / powerFreq)):
            harmonic = (i+1)*powerFreq
            lower = harmonic - 3
            if (harmonic + 3) <= np.max(f):
                upper = harmonic + 3
            elif (harmonic-3) <= np.max(f) <= (harmonic+3):
                upper = np.max(f)
            noise = np.where(np.logical_and(lower<=f, f<=upper))
            line_noise_freq.extend([min(f[noise]), max(f[noise])])
            f = np.delete(f, noise)
            Zxx = np.delete(Zxx, noise, axis=1)

        # print('Removed {} frequency bins'.format(nBins - len(f)))
        # other transformations
        Zxx = np.log(np.abs(Zxx)) + 1e-6

        # plot stft for channel 0
        # plt.pcolormesh(t, f, Zxx[0,:,:], shading='gouraud', cmap='inferno', vmin=-4.5, vmax=9) # use this for human
        # plt.pcolormesh(t, f, Zxx[0, :, :], shading='gouraud', cmap='inferno')
        # plt.title('STFT Magnitude (clip {})'.format(clip))
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.colorbar()
        # plt.show()
        self.broken_axis_plot(t, f, Zxx, line_noise_freq)

        return f, t, Zxx

    def cacheData(self, X, y):
        filename = '{self.phase}_{self.target}.hkl'.format(self=self)  # e.g. ictal_Patient_1.hkl
        filepath = os.path.join(self.settings['cachedir'], filename)
        print('Caching {}...'.format(filename))
        hkl.dump([X, y], filepath, mode='w')
        print('Done caching')

    def apply(self):
        filename = '{self.phase}_{self.target}.hkl'.format(self=self)  # e.g. ictal_Patient_1.hkl
        filepath = os.path.join(self.settings['cachedir'], filename)

        # load preprocessed data from cache (if it exists)
        if os.path.isfile(filepath):
            print('Loading {} from cache...'.format(filename))
            X, y = hkl.load(filepath)
            print('Done loading')

        # preprocess data
        else:
            data = self.read_raw_signal()
            if self.settings['dataset']=='Kaggle2014Pred':
                X, y = self.preprocess(data)

            else:
                # different preprocess steps for other datasets
                X, y = self.preprocess(data)

            print('Caching {}...'.format(filename))
            hkl.dump([X, y], filepath, mode='w')
            print('Done caching')

        return X, y

    def broken_axis_plot(self, x, y, z, skip_y):
        if 'Dog_' in self.target:
            fig, axs = plt.subplots(2, 1, sharex=True)
            fig.subplots_adjust(hspace=0)

            ax1 = axs[0]
            ax2 = axs[1]

            ax1.pcolormesh(x, y, z[0, :, :], shading='gouraud', cmap='inferno', vmax=z[0::].max(), vmin=z[0::].min())
            ax2.pcolormesh(x, y, z[0, :, :], shading='gouraud', cmap='inferno', vmax=z[0::].max(), vmin=z[0::].min())

            # limit view to different portions of the data
            ax1.set_ylim(bottom=skip_y[1])
            ax2.set_ylim(top=skip_y[0])

            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(top=False, labeltop=False)  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            # d = .5
            # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=4,
            #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            kwargs = dict(marker=">", markersize=4,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            # ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

            fig.text(0.5, 0.04, 'Time [sec]', ha='center')
            fig.text(0.04, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')
            norm = cm.colors.Normalize(vmax=z[0::].max(), vmin=z[0::].min())

        if 'Patient_' in self.target:
            ratios = [1]
            ratios.extend(8*[3])
            fig, axs = plt.subplots(9, 1, sharex=True, gridspec_kw={'height_ratios': ratios})
            fig.subplots_adjust(hspace=0)
            # plt.tight_layout()

            for i, ax in enumerate(axs):
                # d = .5
                # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=4,
                #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
                kwargs = dict(marker=">", markersize=4,
                              linestyle="none", color='k', mec='black', mew=1, clip_on=False)

                ax.pcolormesh(x, y, z[0, :, :], shading='gouraud', cmap='inferno', vmax=z[0::].max(), vmin=z[0::].min())
                ax.set_yticks(np.arange(0, 501, 50))

                if i == 0:
                    ax.set_ylim(bottom=skip_y[-1])
                    ax.spines['bottom'].set_visible(False)
                    ax.xaxis.tick_top()
                    ax.tick_params(top=False, labeltop=False)
                    ax.plot(0, 0, transform=ax.transAxes, **kwargs)

                elif i==8:
                    ax.set_ylim(top=skip_y[0])
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.tick_bottom()
                    # ax.plot([0, 1], [1, 1], transform=ax.transAxes, **kwargs)

                else:
                    ax.set_ylim(bottom=skip_y[-2*i - 1], top=skip_y[-2*i])
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.tick_top()
                    ax.tick_params(top=False, labeltop=False)
                    # ax.plot([0, 0, 1, 1], [0, 1, 0, 1], transform=ax.transAxes, **kwargs)
                    ax.plot(0, 0, transform=ax.transAxes, **kwargs)

            # fig.axes[0].set_xlabel('Time [sec]')
            fig.text(0.5, 0.04, 'Time [sec]', ha='center')
            fig.text(0.04, 0.5, 'Frequency [Hz]', ha='center', va='center', rotation='vertical')
            norm = cm.colors.Normalize(vmax=9, vmin=-4.5)

        cmap = cm.ScalarMappable(norm, 'inferno')
        fig.colorbar(cmap, ax=axs.flat)
        plt.show()