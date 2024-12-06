import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from tqdm import tqdm
import librosa
import re
import itertools

def metadata_to_label(data_dirs):
    meta2label = {}
    label2meta = {}
    label = 0
    for data_dir in data_dirs:
        machine = data_dir.split('/')[-2]
        id_list = get_machine_id_list(data_dir)
        for id_str in id_list:
            meta = machine + '-' + id_str
            meta2label[meta] = label
            label2meta[label] = meta
            label += 1
    return meta2label, label2meta


def get_machine_id_list(data_dir):
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('section_[0-9][0-9]', ext_id) for ext_id in get_filename_list(data_dir)])
    )))
    return machine_id_list


def get_filename_list(dir_path, pattern='*', ext='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob(file_path_pattern))
        filename_list += files
    return filename_list



class DCASE2022Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs, model_save_dir, is_train=True, need_convert_mel=True, use_gmm=False):
        '''
        is_train=True：表示现在在训练阶段，训练阶段需要获取对应section的归一化参数并保存下来。而且__getitem__需要返回，归一化后的mel谱，以及对应的类别标签
        is_train=False：表示在测试阶段，归一化参数是训练集的，__getitem__需要返回mel、正异常、domain、以及类别
        need_convert_mel=True：表名不存在归一化后的.npy文件，需要重新生成
        '''
        super().__init__()
        
        self.is_train = is_train
        self.use_gmm = use_gmm
        self.model_save_dir = model_save_dir
        if need_convert_mel:
            self.all_filename_list = self.preprocessing(data_dirs)  
        else:
            self.all_filename_list = []
            for data_dir in data_dirs:
                self.all_filename_list.extend(self.get_filename_list(data_dir, ext='npy'))
        
        if is_train and use_gmm == False:  
            self.meta2label, self.label2meta = metadata_to_label(data_dirs)
            torch.save((self.meta2label, self.label2meta), os.path.join(self.model_save_dir, 'section_mapping'))
        else:  
            self.meta2label, self.label2meta = torch.load(os.path.join(self.model_save_dir, 'section_mapping'))

            
    def __getitem__(self, index):
        mel = np.load(self.all_filename_list[index][:-4] + '.npy')
        machine = self.all_filename_list[index].split('/')[-3]
        id_str = re.findall('section_[0-9][0-9]', self.all_filename_list[index])[0]
        class_label = self.meta2label[machine+'-'+id_str]
        if self.is_train or self.use_gmm:
            return mel, class_label
        else:
            if 'normal' in self.all_filename_list[index]:
                label = 0
            else:
                label = 1
            
            if 'source' in self.all_filename_list[index]:
                domain = 0
            else:
                domain = 1
            return mel, class_label, label, domain
        
        
    def __len__(self):
        return len(self.all_filename_list)
    
    def preprocessing(self, data_dirs):
        all_filename_list = []
        for data_dir in data_dirs:  
            machine = data_dir.split('/')[-2]  
            
            filenames = self.get_filename_list(data_dir, ext='wav') 
            all_filename_list.extend(filenames)
            machine_id_list = sorted(list(set(
                itertools.chain.from_iterable([re.findall('section_[0-9][0-9]', ext_id) for ext_id in self.get_filename_list(data_dir)])
            )))
            spec = [[], [], []]
            all_spec = dict(zip(machine_id_list, spec))  
            
            for filename in tqdm(filenames, desc=f'Generating Data: {machine}'):
                section_id = re.findall('section_[0-9][0-9]', filename)
                
                mel = self.makemel(filename)
                if self.is_train:        
                    all_spec[section_id[0]].append(mel)
            if self.is_train:  
                mean_std_max_min = {}  # {'section_00': [mean, std, max, min], 'section_01': [mean, std, max, min]}
                for section in all_spec.keys():
                    spec = np.array(all_spec[section])
                    mean, std, max, min = self.cal_scale_param(spec)
                    mean_std_max_min[section] = [mean, std, max, min]
                if not os.path.exists(os.path.join(self.model_save_dir, 'scale_params')):
                    os.mkdir(os.path.join(self.model_save_dir, 'scale_params'))
                torch.save(mean_std_max_min, os.path.join(self.model_save_dir, f'scale_params/{machine}_mean_std_max_min.npy'))
                print(f'generating {machine} scale params done')
            else:
                mean_std_max_min = torch.load(os.path.join(self.model_save_dir, f'scale_params/{machine}_mean_std_max_min.npy'))
            
            for filename in tqdm(filenames, desc=f'scaling {machine} data'):
                mel = np.load(filename[:-4] + '.npy')
                section_id = re.findall('section_[0-9][0-9]', filename)
                mel = self.normalize_data(mel, mean_std_max_min[section_id[0]])
                np.save(filename[:-4] + '.npy', mel)
        return all_filename_list
    
    def makemel(self, filename):
        wav, _ = librosa.load(filename, sr=16000)
        wav = wav[: 160000]
        D = np.abs(librosa.stft(wav,hop_length=626)) ** 2  
        S = librosa.feature.melspectrogram(S=D, n_mels=256) 
        mel = librosa.power_to_db(S, ref=np.max)
        mel = np.expand_dims(mel, axis=0)
        np.save(filename[:-4] + '.npy', mel)
        return mel
                
    def cal_scale_param(self, spec):
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / std
        max = spec.max()
        min = spec.min()
        return mean, std, max, min
    
    def normalize_data(self, mel, mean_std_max_min):
        mean, std, max, min = mean_std_max_min
        mel = (mel - mean) / std
        mel = (mel - min) / (max - min)
        mel = mel * 2 - 1
        return mel
    
    def get_filename_list(self, dir_path, pattern='*', ext='*'):
        """
        find all extention files under directory
        :param dir_path: directory path
        :param ext: extention name, like wav, png...
        :param pattern: filename pattern for searching
        :return: files path list
        """
        filename_list = []
        for root, _, _ in os.walk(dir_path):
            file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
            files = sorted(glob(file_path_pattern))
            filename_list += files
        return filename_list
    