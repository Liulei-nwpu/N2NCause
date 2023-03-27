import numpy as np

import sys,os
# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)



class DatasetExperiment:
    
    def __init__(self, dev_ratio=0.1, test_ratio=0.1):
        self.data_dir = self._data_path()
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
    
    def train_set(self):
        raise NotImplementedError
    
    def train_set_pairs(self):
        raise NotImplementedError
    
    def train_labels(self):
        raise NotImplementedError
    
    def dev_set(self):
        raise NotImplementedError
    
    def dev_set_pairs(self):
        raise NotImplementedError
    
    def dev_labels(self):
        raise NotImplementedError
    
    def test_set(self):
        raise NotImplementedError
    
    def test_set_pairs(self):
        raise NotImplementedError
    
    def test_labels(self):
        raise NotImplementedError
    
    def _data_path(self):
        raise NotImplementedError


class Dataset:
    
    def __init__(self, vectorizer, dataset, batch_size):
        self.train_sen1, self.train_sen2 = vectorizer.vectorize_2d(dataset.train_set_pairs())
        self.dev_sen1, self.dev_sen2 = vectorizer.vectorize_2d(dataset.dev_set_pairs())
        self.test_sen1, self.test_sen2 = vectorizer.vectorize_2d(dataset.test_set_pairs())
        self.num_tests = len(dataset.test_set())
        self._train_labels = dataset.train_labels()
        self._dev_labels = dataset.dev_labels()
        self._test_labels = dataset.test_labels()
        self.__shuffle_train_idxs = range(len(self._train_labels))
        self.num_batches = len(self._train_labels) // batch_size
        print(self.num_batches)
    
    def train_instances(self, shuffle=False):
        if shuffle:
            self.__shuffle_train_idxs = np.random.permutation(range(len(self.__shuffle_train_idxs)))
            self.train_sen1 = self.train_sen1[self.__shuffle_train_idxs]
            self.train_sen2 = self.train_sen2[self.__shuffle_train_idxs]
            self._train_labels = self._train_labels[self.__shuffle_train_idxs]
        return self.train_sen1, self.train_sen2
    
    def train_labels(self):
        return self._train_labels
    
    def test_instances(self):
        return self.test_sen1, self.test_sen2
    
    def test_labels(self):
        return self._test_labels
    
    def dev_instances(self):
        return self.dev_sen1, self.dev_sen2, self._dev_labels
    
    def num_dev_instances(self):
        return len(self._dev_labels)
    
    def pick_train_mini_batch(self):
        train_idxs = np.arange(len(self._train_labels))
        np.random.shuffle(train_idxs)
        train_idxs = train_idxs[:self.num_dev_instances()]
        mini_train1 = self.train_sen1[train_idxs]
        mini_train2 = self.train_sen2[train_idxs]
        mini_labels = self._train_labels[train_idxs]
        return mini_train1, mini_train2, mini_labels
        
    def __str__(self):
        return 'Dataset properties:\n ' \
               'Number of training instances: {}\n ' \
               'Number of dev instances: {}\n ' \
               'Number of test instances: {}\n' \
            .format(len(self._train_labels), len(self._dev_labels), len(self._test_labels))
