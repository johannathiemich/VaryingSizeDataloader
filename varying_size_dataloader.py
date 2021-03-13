from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch
from baumbauen.utils import pad_collate_fn
from torch.utils.data.sampler import BatchSampler
from baumbauen.utils import SimilarSizeSampler
from baumbauen.utils import BucketingSampler


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

