import numpy as np


class CnnData(object):

    def __init__(self, data_config):
        self.d = data_config

        # full image cnn
        self.cnn_full_img_train = np.array([])
        self.cnn_full_img_val = np.array([])
        self.cnn_full_img_test = np.array([])

        # full image + regions cnn
        self.cnn_region_train = np.array([])
        self.cnn_region_val = np.array([])
        self.cnn_region_test = np.array([])

        return

    def set_cnn_full_img_split(self, split='test'):
        if split == 'train':
            fname = self.d['cnn_full_img_path_train']
            self.cnn_full_img_train = np.loadtxt(fname, delimiter=',')
        elif split == 'val':
            fname = self.d['cnn_full_img_path_val']
            self.cnn_full_img_val = np.loadtxt(fname, delimiter=',')
        elif split == 'test':
            fname = self.d['cnn_full_img_path_test']
            self.cnn_full_img_test = np.loadtxt(fname, delimiter=',')
        else:
            raise ValueError("only train, val and test splits supported")
        return

    def set_cnn_regions_split(self, split):
        if split == 'train':
            fname = self.d['cnn_regions_path_train']
            self.cnn_region_train = np.loadtxt(fname, delimiter=',')
        elif split == 'val':
            fname = self.d['cnn_regions_path_val']
            self.cnn_region_val = np.loadtxt(fname, delimiter=',')
        elif split == 'test':
            fname = self.d['cnn_regions_path_test']
            self.cnn_region_test = np.loadtxt(fname, delimiter=',')
        else:
            raise ValueError("only train, val and test splits supported")
        return

    def get_cnn_regions_split(self, split):
        if split == 'train':
            if self.cnn_region_train.size == 0:
                self.set_cnn_regions_split(split)
            return self.cnn_region_train
        elif split == 'val':
            if self.cnn_region_val.size == 0:
                self.set_cnn_regions_split(split)
            return self.cnn_region_val
        elif split == 'test':
            if self.cnn_region_test.size == 0:
                self.set_cnn_regions_split(split)
            return self.cnn_region_test
        else:
            raise ValueError("only train, val and test splits supported")

    def get_cnn_full_img_split(self, split):
        if split == 'train':
            if self.cnn_full_img_train.size == 0:
                self.set_cnn_full_img_split(split)
            return self.cnn_full_img_train
        elif split == 'val':
            if self.cnn_full_img_val.size == 0:
                self.set_cnn_full_img_split(split)
            return self.cnn_full_img_val
        elif split == 'test':
            if self.cnn_full_img_test.size == 0:
                self.set_cnn_full_img_split(split)
            return self.cnn_full_img_test
        else:
            raise ValueError("only train, val and test splits supported")

