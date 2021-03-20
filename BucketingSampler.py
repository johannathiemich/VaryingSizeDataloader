import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import random
import torch
import operator
import math
from itertools import chain
from baumbauen.utils import pad_collate_fn
import copy
from torch.utils.data.sampler import BatchSampler


class BucketingSampler(Sampler):
    """
    The BucketingSampler draws a mega batch of data samplers, then sorts this mega batch by length and yields those indices.
    """
    #small parts from https://github.com/shenkev/Pytorch-Sequence-Bucket-Iterator/blob/master/torch_sampler.py
    def __init__(self, data_source, batch_size = 64, drop_last = False, num_bins=5, replacement=False):
        """
        Initializing a SimilarSizeSampler object
        data_source: data to draw samples from
        replacement: whether to draw the samples with or without replacement
        batch_size: number of batches to group together
        drop_last: whether to drop the last batch (that potentially has a smaller batch size than the rest
                    of the batches when len(data_source) % batch_size != 0
        num_bins: number of bins to draw from data; the total number of samples in a mega batch will be num_bins*batch_size 
        """
        self.data_source = [item[0] for item in data_source]
        self.ind_len_arr = [(i, item.shape[0]) for i, item in enumerate(self.data_source)]

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_bins = num_bins
        self.replacement = replacement

        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

    def __iter__(self):
        """
        This function returns the indices (one after another) that will be used by the BatchSampler in order to create the DataLoader
        """
        iter_list = []
        ind_len_copy = copy.deepcopy(self.ind_len_arr)

        ind_len_copy.sort(key=operator.itemgetter(1))

        if self.replacement:
            num_samples = 0
            while num_samples < self.__len__():
                samples = random.sample(ind_len_copy, k = self.num_bins*self.batch_size)
                samples.sort(key=operator.itemgetter(1))
                for i in range(self.num_bins):
                    new_batch = [item[0] for item in samples[(i * self.batch_size): (i + 1) * self.batch_size]]
                    random.shuffle(new_batch)
                    assert len(new_batch) == self.batch_size
                    iter_list.append(new_batch)

                num_samples += len(samples)
                assert(len(samples) == self.batch_size * self.num_bins)

            if num_samples - self.__len__() > self.batch_size:
                num_batches = math.floor((num_samples - self.__len__()) / self.batch_size)
                iter_list = iter_list[:-num_batches]
                num_samples -= num_batches * self.batch_size

            if (num_samples > self.__len__()):
                last_batch = iter_list[-1]
                iter_list = iter_list[:-1]
                iter_list.append(last_batch[:len(last_batch) - (num_samples - self.__len__())])
                num_samples -= len(iter_list[0]) - len(iter_list[-1])
            assert (num_samples == self.__len__())

            rest = self.__len__() - num_samples
            if rest > 0:
                rest_samples = random.sample(ind_len_copy, k = rest)
                rest_samples.sort(key=operator.itemgetter(1))
                iter_list.append(rest_samples)

        else:
            while len(ind_len_copy) > 0:
                if len(ind_len_copy) < self.num_bins * self.batch_size:
                    no_batches = math.floor(len(ind_len_copy) / self.batch_size)
                    for i in range(no_batches):
                        new_batch = [item[0] for item in ind_len_copy[(i * self.batch_size) : (i + 1) * self.batch_size]]
                        random.shuffle(new_batch)
                        iter_list.append(new_batch)
                        assert len(new_batch) > 0

                    rest_batch = [item[0] for item in ind_len_copy[no_batches * self.batch_size:len(self.ind_len_arr)]]
                    random.shuffle(rest_batch)
                    iter_list.append(rest_batch)
                    ind_len_copy = []
                else:
                    #draw a mega batch and sort it 
                    samples = random.sample(ind_len_copy, k=self.num_bins*self.batch_size)
                    for s in samples:
                        ind_len_copy.remove(s)

                    samples.sort(key=operator.itemgetter(1))

                    for i in range(self.num_bins):
                        #create smaller batches from mega batch
                        new_batch = [item[0] for item in samples[(i * self.batch_size) : (i + 1) * self.batch_size]]
                        #shuffle the samples inside each mini batch
                        random.shuffle(new_batch)
                        assert len(new_batch) == self.batch_size
                        iter_list.append(new_batch)

        assert len(iter_list) > 0

        if len(iter_list[-1]) != self.batch_size and self.drop_last:
            iter_list = iter_list[:-1]

        for item in iter_list:
            for i in item:
                yield i

    def __len__(self):
        """
        This function returns the number of indices that will be returned. 
        The result depends on the length of the data source, the batch size and whether the last batch will be dropped or not
        """
        if self.drop_last:
            num_elem = len(self.data_source) - (len(self.data_source) % self.batch_size)
            return len(self.data_source) - (len(self.data_source) % self.batch_size)
        else:
            return len(self.data_source)


