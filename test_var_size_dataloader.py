import torch
import unittest

from baumbauen.utils import SimilarSizeSampler
from baumbauen.utils import BucketingSampler
from torch.utils.data.sampler import  BatchSampler

class VarSizeSamplerTest(unittest.TestCase):
    def test_correct_length(self):
        input_data = torch.randn((103, 1, 4, 10))

        buck_sampler_no_drop_last = BucketingSampler(input_data, batch_size=20, drop_last=False, num_bins=3, replacement=False)
        sim_sampler_no_drop_last = SimilarSizeSampler(input_data, batch_size=20, drop_last=False, replacement=False)

        assert(buck_sampler_no_drop_last.__len__() == 103)
        assert(sim_sampler_no_drop_last.__len__() == 103)

        counter = 0
        for _ in buck_sampler_no_drop_last.__iter__():
            counter = counter + 1
        assert(counter == 103)

        counter = 0
        for _ in buck_sampler_no_drop_last.__iter__():
            counter = counter + 1
        assert(counter == 103)

        buck_sampler_drop_last = BucketingSampler(input_data, batch_size=20, drop_last=True, num_bins=3, replacement=False)
        sim_sampler_drop_last = SimilarSizeSampler(input_data, batch_size=20, drop_last=True, replacement=False)

        assert(buck_sampler_drop_last.__len__() == 100)
        assert(sim_sampler_drop_last.__len__() == 100)

        counter = 0
        for _ in buck_sampler_drop_last.__iter__():
            counter = counter + 1
        assert(counter == 100)

        counter = 0
        for _ in sim_sampler_drop_last.__iter__():
            counter = counter + 1
        assert(counter == 100)

    def test_batch_sampler(self):
        input_data = torch.randn((103, 1, 4, 10))

        buck_sampler_no_drop_last = BucketingSampler(input_data, batch_size=20, drop_last=False, num_bins=3, replacement=False)
        buck_batch_sampler_no_drop_last = BatchSampler(buck_sampler_no_drop_last, batch_size=20, drop_last=False)

        sim_sampler_no_drop_last = SimilarSizeSampler(input_data, batch_size=20, drop_last=False, replacement=False)
        sim_batch_sampler_no_drop_last = BatchSampler(sim_sampler_no_drop_last, batch_size=20, drop_last=False)

        assert(sim_batch_sampler_no_drop_last.__len__() == 6)
        assert(buck_batch_sampler_no_drop_last.__len__() == 6)

        counter = 0
        for _ in buck_batch_sampler_no_drop_last.__iter__():
            counter = counter + 1
        assert(counter == 6)

        counter = 0
        for _ in sim_batch_sampler_no_drop_last.__iter__():
            counter = counter + 1
        assert (counter == 6)

        buck_sampler_drop_last = BucketingSampler(input_data, batch_size=20, drop_last=True, num_bins=3, replacement=False)
        buck_batch_sampler_drop_last = BatchSampler(buck_sampler_drop_last, batch_size=20, drop_last=True)

        sim_sampler_drop_last = SimilarSizeSampler(input_data, batch_size=20, drop_last=True, replacement=False)
        sim_batch_sampler_drop_last = BatchSampler(sim_sampler_drop_last, batch_size=20, drop_last=True)

        assert(sim_batch_sampler_drop_last.__len__() == 5)
        assert(buck_batch_sampler_drop_last.__len__() == 5)

        counter = 0
        for _ in buck_batch_sampler_drop_last.__iter__():
            counter = counter + 1
        assert (counter == 5)

        counter = 0
        for _ in sim_batch_sampler_drop_last.__iter__():
            counter = counter + 1
        assert (counter == 5)


if __name__ == '__main__':
    unittest.main()