from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import librosa
import torch


class UrbanSound8K(Dataset):
    def __init__(self, dataset_root, unified_sample_rate, unified_sec, mode, train_ratio):
        self.root = dataset_root
        self.sample = []
        item_list = pd.read_csv(os.path.join(self.root, 'UrbanSound8K.csv'))
        for index, item in item_list.iterrows():
            # fetch necessary information of one sample
            file_name = item['slice_file_name']
            label = item['classID']
            start = item['start']
            end = item['end']
            folder = item['fold']
            audio_path = os.path.join(self.root, 'fold' + str(folder), file_name)

            # load and preprocess
            audio_wave, sample_rate = librosa.load(audio_path)
            audio_wave = audio_wave[int(start * sample_rate):int(end * sample_rate)]
            audio_wave = librosa.resample(audio_wave, orig_sr=sample_rate, target_sr=unified_sample_rate)

            # padding the audio_wave to 4 sec
            desired_length = unified_sample_rate * unified_sec
            current_length = len(audio_wave)

            if current_length < desired_length:
                # Pad the audio waveform with zeros at the end
                audio_wave = np.pad(audio_wave, (0, desired_length - current_length), 'constant')

            self.sample.append((torch.from_numpy(audio_wave), int(label)))

        # split the dataset
        num_train_samples = int(len(self.sample) * train_ratio)
        if mode == 'train':
            self.sample = self.sample[:num_train_samples]
        elif mode == 'test':
            self.sample = self.sample[num_train_samples:]

        # end

    def __getitem__(self, index):
        return self.sample[index]

    def __len__(self):
        return len(self.sample)


dataset_train = UrbanSound8K(dataset_root='../data/UrbanSound8k', unified_sample_rate=22050, unified_sec=4,
                             mode='train',
                             train_ratio=0.8)

dataset_test = UrbanSound8K(dataset_root='../data/UrbanSound8k', unified_sample_rate=22050, unified_sec=4,
                            mode='test',
                            train_ratio=0.8)

train_loader = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=128, shuffle=False)

"""

for i, data in enumerate(train_loader):
    inputs, label = data
    print(len(inputs), len(label))
    break
    
# print -> 128 128
"""
