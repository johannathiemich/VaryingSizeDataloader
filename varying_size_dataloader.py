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


class TestDataset(Dataset):
    def __init__(self, random_lengths=False, num_items=100):
        self.num_items = num_items
        self.data = []
        self.labels = []
        if not random_lengths:
            self.data = [
                torch.rand((50, 1, 10))
            ]

        else:
            for i in range(num_items):
                len = random.randrange(2, 60)
                self.data.append(torch.rand((50, len, 6)))
                self.labels.append(torch.rand(50, len, 1))
        print("initialized")

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class SimilarSizeSampler(Sampler):
    def __init__(self, data_source, replacement=False, batch_size=64, drop_last=False):
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
        #max_len = max([item[1] for item in self.ind_len_arr])
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

            assert [len(item) == self.batch_size for item in iter_list]

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
        if self.drop_last:
            num_elem = len(self.data_source) - (len(self.data_source) % self.batch_size)
            return len(self.data_source) - (len(self.data_source) % self.batch_size)
        else:
            return len(self.data_source)



class BucketingSampler(Sampler):
    #small parts from https://github.com/shenkev/Pytorch-Sequence-Bucket-Iterator/blob/master/torch_sampler.py
    def __init__(self, data_source, batch_size = 64, drop_last = False, num_bins=5, replacement=False):
        self.data_source = [item[0] for item in data_source]
        self.ind_len_arr = [(i, item.shape[0]) for i, item in enumerate(self.data_source)]

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_bins = num_bins
        self.replacement = replacement

        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

    def __iter__(self):
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
                    samples = random.sample(ind_len_copy, k=self.num_bins*self.batch_size)
                    for s in samples:
                        ind_len_copy.remove(s)

                    samples.sort(key=operator.itemgetter(1))

                    for i in range(self.num_bins):
                        new_batch = [item[0] for item in samples[(i * self.batch_size) : (i + 1) * self.batch_size]]
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
        if self.drop_last:
            num_elem = len(self.data_source) - (len(self.data_source) % self.batch_size)
            return len(self.data_source) - (len(self.data_source) % self.batch_size)
        else:
            return len(self.data_source)

def main():
    dataset = TestDataset(random_lengths=True)
    sim_size_sampler = SimilarSizeSampler(dataset, replacement=False, batch_size=3)
    bucketing_sampler = BucketingSampler(dataset, batch_size=3)

    # Prep dataloaders
    sim_dataloader = DataLoader(dataset,
                                batch_size=1,
                                batch_sampler=sim_size_sampler,
                                num_workers=0,
                                collate_fn=pad_collate_fn,
                                drop_last=False,
                                pin_memory=False)
    buck_dataloader = DataLoader(dataset,
                                 batch_size=1,
                                 batch_sampler=bucketing_sampler,
                                 num_workers=0,
                                 collate_fn=pad_collate_fn,
                                 drop_last=False,
                                 pin_memory=False)
    
    # Prep Batch samplers
    batch_sampler_sim = BatchSampler(sim_size_sampler, batch_size=3, drop_last=False)
    batch_sampler_bucket = BatchSampler(bucketing_sampler, batch_size=3, drop_last=False)

    for batch in BatchSampler:
        print(batch)


if __name__ == "__main__":
    main()

