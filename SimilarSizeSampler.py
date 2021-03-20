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


class SimilarSizeSampler(Sampler):
    """
    The SimilarSizeSampler sorts the data samples by length first and then forms the batches, so that similar sized samples are always in the same batch
    """
    
    def __init__(self, data_source, replacement=False, batch_size=64, drop_last=False):
        """
        Initializing a SimilarSizeSampler object
        data_source: data to draw samples from
        replacement: whether to draw the samples with or without replacement
        batch_size: number of batches to group together
        drop_last: whether to drop the last batch (that potentially has a smaller batch size than the rest
                    of the batches when len(data_source) % batch_size != 0
        """
        #data source shape is (N, L, C)
        self.data_source = [item[0] for item in data_source]

        self.batch_size = batch_size
        self.replacement = replacement
        self.drop_last = drop_last
        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

        #create (ind, len) list:
        self.ind_len_arr = [(i, item.shape[0]) for i, item in enumerate(self.data_source)]

        # create probability distribution function
        # this probability distribution is only necessary for the sampler with replacement
        if self.replacement:
            max_len = 60
            self.pdf = [0] * (max_len + 1)

            #counting how often each length appears and creating a dictionary with (key, value) = (length, indices in dataset)
            self.len_buckets = dict()
            for idx, len in self.ind_len_arr:
                self.pdf[len] += 1
                if (len) in self.len_buckets.keys():
                    self.len_buckets[len].append(idx)
                else:
                    self.len_buckets[len] = [idx]

            missing_keys = set(range(max_len)) - set(self.len_buckets.keys())
            for key in missing_keys:
                self.len_buckets[int(key)] = []

            #normalize probabilities
            self.pdf = [count / (sum(self.pdf) * 1.0) for count in self.pdf]
            assert math.isclose(sum(self.pdf), 1.0, rel_tol=1e-09, abs_tol=0.0)

            #create distribution
            self.distr = Categorical(torch.tensor(self.pdf))

    def __iter__(self):
        """
        This function returns the indices (one after another) that will be used by the BatchSampler in order to create the DataLoader
        """
        
        iter_list = []

        if self.replacement:
            batches = torch.tensor(0)
            for i in range(0, len(self.data_source), self.batch_size):
                # while iterating over Dataset: draw length parameter from probability distribution calculated in init
                #sampled_len = self.distr.sample()
                sampled_len = random.sample(self.data_source).shape[0]
                sampled_batch_indices = [self.len_buckets[int(sampled_len)]]

                #if there are not enough samples of the required length in the dataset, additionally draw samples of
                # similar length (start with "surrounding" length and gradually increase distance to required length
                if len(sampled_batch_indices[0]) < self.batch_size:
                    pos_difference = 1
                    neg_difference = 1

                    search_left = True
                    search_right = True
                    while len(sampled_batch_indices) < self.batch_size:
                        if search_right:
                            sampled_batch_indices.append(self.len_buckets[int(sampled_len + pos_difference)])
                        if search_left:
                            sampled_batch_indices.append(self.len_buckets[int(sampled_len - neg_difference)])
                        pos_difference += 1
                        neg_difference += 1
                        #make sure we are not stepping out of bounds; if either min or max length for samples sizes is reached,
                        # keep only exploring options in one direction
                        search_right = (sampled_len + pos_difference) < len(self.pdf)
                        search_left = (sampled_len - neg_difference) > 0

                        assert search_right or search_left

                #filtering out the indices which contain empty lists
                sampled_batch_indices = [item for item in sampled_batch_indices if len(item) > 0]
                #flatten list
                sampled_batch_indices = list(chain.from_iterable(sampled_batch_indices))
                #randomly picking batch_size elements from list
                sampled_batch_indices = random.choices(sampled_batch_indices, k=self.batch_size)

                random.shuffle(sampled_batch_indices)
                iter_list.append(sampled_batch_indices)
        else:
            ind_len_copy = copy.deepcopy(self.ind_len_arr)
            ind_len_copy.sort(key=operator.itemgetter(1))

            for i in range(0, self.__len__(), self.batch_size):
                new_batch = [item[0] for item in ind_len_copy[i:i + (self.batch_size)]]
                random.shuffle(new_batch)
                iter_list.append(new_batch)

            #assert [len(item) == self.batch_size for item in iter_list]

        #checking whether to drop the last batch or not
        if len(iter_list[-1]) != self.batch_size and self.drop_last:
            iter_list = iter_list[:-1]

        iter_list_copy = copy.deepcopy(iter_list)
        iter_list_copy = iter_list_copy[:-1]
        random.shuffle(iter_list_copy)
        iter_list_copy.append(iter_list[-1])
        for item in iter_list_copy:
            for it in item:
                yield it

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
