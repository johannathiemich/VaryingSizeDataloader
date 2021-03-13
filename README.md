# VaryingSizeDataloader

Practical course "Data Management and Data Analysis" at Karlsruhe Institute of Technology during the Winter Semester 2020/21.
Supervisor: James Kahn
The goal was to implement two samplers that are able to deal with data of varying sizes, arranging them in batches so that the overhead
introduced by padding the data to a uniform length is minimized.
Two different data loaders were implemented: SimilarSizeSampler and BucketingSampler. Both are implemented for drawing with and without replacement 
as well as with and without dropping the last batch (if its size is smaller than the batch size)

##SimilarSizeSampler
The SimilarSizeSampler sorts the data samples by length first and then forms the batches, so that similar sized samples are always in the same batch

##BucketingSampler
The BucketingSampler draws a mega batch of data samplers, then sorts this mega batch by length and yields those indices. 

##TestDataset
I provided a simple TestDataset that creates tensors with random numbers that can be used to test the samplers.

##Unittest
Prvoided in this repository are also a couple of short unit tests to ensure the samplers are working correctly.
