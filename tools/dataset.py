from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
from torchvision.datasets.folder import default_loader
import torch.utils.data as Data
import cv2 as cv
import torchvision.transforms as transforms
class datasets(Data.Dataset):
    """
    VeRi
    Reference:
    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
    International Conference on Multimedia and Expo. (2016) accepted.

    Dataset statistics:
    # identities: 776 vehicles(576 for training and 200 for testing)
    # images: 37778 (train) + 11579 (query)
    """
    def __init__(self, dataset_root='',split='train',size = 128,**kwargs):
        if not split in ['train', 'test','pred']:
            raise Exception('Invalid dataset split.')
        self.dataset_dir = dataset_root
        self.split = split
        if split == 'train':
            self.data_path = osp.join(self.dataset_dir, self.split)
            self.train = self.process_dir(self.data_path)
            self.path=[i[0] for i in self.train]
            self.label=[i[1] for i in self.train]
            self.idx=[i[2] for i in self.train]
        elif split == 'test':
            self.data_path = osp.join(self.dataset_dir, self.split)
            self.test = self.process_dir(self.data_path)
            self.path = [i[0] for i in self.test]
            self.label = [i[1] for i in self.test]
            self.idx=[i[2] for i in self.test]
        elif split == 'pred': 
            self.data_path = osp.join(self.dataset_dir, self.split)
            # self.test = self.process_dir(self.data_path)
            self.path = glob.glob(osp.join(self.data_path, '*.png'))
            self.pred = [i.split('/')[-1] for i in self.path]
            self.idx=[i.split('.')[0] for i in self.pred]


        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.loader= default_loader
        self.transform = transforms.Compose([
            transforms.Resize(int(size / 0.875)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])

        self.check_before_run()


        # if verbose:
        #     print('=> VeRi loaded')
            # self.print_dataset_statistics(train, query, gallery)

    def __getitem__(self, index):
        if self.split == 'train' or self.split == 'test':
            path = self.path[index]
            label = self.label[index]
            idx = self.idx[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, label ,idx

        elif self.split == 'pred':
            path = self.path[index]
            idx = self.idx[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, idx

    def __len__(self):
        return len(self.path)
        
    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.data_path):
            raise RuntimeError('"{}" is not available'.format(self.data_path))


    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')
        dataset=[]
        # pid_container = set()
        for img_path in img_paths:
            name = img_path.split('/')[-1]
            label = self.label(name.split('_')[0])
            ids = name.split('_')[1]

            dataset.append((img_path, label, ids ))
        return dataset

    def label(self,label):
        if label == 'CN':
            x = 0
        else:
            x = 1
        return x

