from torch.utils.data import Dataset
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import math
import pickle
import os


"""
This class is used for creating random numpy arrays where the size of one dimension follows one of 5 possible distributions
(Gaussian, Exponential_Right, Exponential_Left, Random, Equal)
Also the numpy arrays are saved to files on the disk.
This was used to create dummy data for evaluating the data samplers.
"""
class TestDatasetGenerator:
    def __init__(self, num_items=100, distribution="Gaussian", min_length=2, max_length=60, train_events=1, feature_dim=4, root_dir="./"):
        """
        num_items: number of numpy files to create
        distribution: the distribution that the length of the drawn samples will follow
        min_length: minimum size of varying dimension
        max_length: maximum size of varying dimension
        train_events: number of arrays per numpy file
        root_dir: where to save the numpy array files to
        """
        self.num_items = num_items
        self.data = []
        self.labels = []
        self.distribution = distribution if distribution in ["Gaussian", "Exponential_Right", "Exponential_Left", "Random", "Equal"] else "Random"
        #checking whether chosen min and max lengths make sense
        if min_length < max_length:
            self.min_length = min_length
            self.max_length = max_length
        else:
            print("Min length of samples must be smaller than max length")
            self.min_length = 2
            self.max_length = 60
        self.train_events = train_events
        self.features_dim = feature_dim

        if os.path.isdir(root_dir):
            self.root_dir = root_dir
            if not os.path.isdir(os.path.join(self.root_dir, self.distribution)):
                #creating the directory to save the files to if it does not exist yet
                os.mkdir(os.path.join(self.root_dir, self.distribution))
            self.save_dir = os.path.join(self.root_dir, self.distribution)
        
        # determining the distibution to use
        if self.distribution == "Gaussian":
            self.create_gaussian()
        elif self.distribution == "Random":
            self.create_random()
        elif self.distribution == "Equal":
            self.create_equal()
        elif self.distribution == "Exponential_Right":
            self.create_exponential_right()
        elif self.distribution == "Exponential_Left":
            self.create_exponential_left()
        else: 
            #if distribution does not match any of the previous ones, use Gaussian
            print("The chosen data distribution is invalid. Using Gaussian distribution instead.")
            self.create_gaussian()

        assert len(self.length_list) == self.num_items
        self.save_length_list()
        
        #these lines can be uncommented if you want to create numpy files from a length list (list that contains the different lengths
        #of the arrays)
        #self.length_list = []
        #self.read_length_list(path="/media/johanna/Volume/Studium/Semester_4/Datamanagement/data/Gaussian_1000/length_list_file")
        #print(self.length_list)

        self.visualize_distribution(verbose=True)
        self.create_tensors()

    def create_gaussian(self):
        """
        This function creates a list of lengths, the lengths are drawn from a Gaussian distribution.
        Samples are drawn until we have enough samples whose length is within [min_range, max_range]
        """
        center = (int)((self.max_length - self.min_length) / 2) + self.min_length
        scale = math.sqrt(((self.max_length - self.min_length) / 2)) + 4

        # from https://stackoverflow.com/questions/16471763/generating-numbers-with-gaussian-function-in-a-range-using-python
        length_list = []
        for _ in range(self.num_items):
            a = np.random.default_rng().normal(center, scale, 1)
            # if drawn length is smaller than desired min length or greater than max length draw the sample again
            while a < self.min_length or a > self.max_length + 1:
                a = random.gauss(center, scale)
            length_list.append(int(a))

        print("max is ", max(length_list))
        print("min is ", min(length_list))
        self.length_list = length_list

    def create_exponential_right(self):
        """
        This function creates a list of lengths, the lengths are drawn from a Exponential Right distribution.
        That is a distribution where most lengths are large, and only few are small.
        Samples are drawn until we have enough samples whose length is within [min_range, max_range]
        """
        scale = int((self.max_length - self.min_length) / 2) - 4

        length_list = []
        for _ in range(self.num_items):
            a = np.random.default_rng().exponential(scale, size=1)
            while a < self.min_length or a > self.max_length + 1:
                a = np.random.exponential(scale, size=1)
            length_list.append(int(a))

        length_list = [-it + self.max_length + self.min_length for it in length_list]
        print("max is ", max(length_list))
        print("min is ", min(length_list))
        self.length_list = length_list

    def create_exponential_left(self):
        """
        This function creates a list of lengths, the lengths are drawn from a Exponential Left distribution.
        That is a distribution where most lengths are small, and only few are large.
        Samples are drawn until we have enough samples whose length is within [min_range, max_range]
        """
        scale = int((self.max_length - self.min_length) / 2) - 4

        length_list = []
        for _ in range(self.num_items):
            a = np.random.default_rng().exponential(scale, size=1)
            while a < self.min_length or a > self.max_length + 1:
                a = np.random.exponential(scale, size=1)
            length_list.append(int(a))

        print("max is ", max(length_list))
        print("min is", min(length_list))
        self.length_list = length_list

    def create_equal(self):
        """
        This function creates a list of lengths, the lengths are drawn from an Equal distribution.
        That is a distribution where all lengths are the same.
        Samples are drawn until we have enough samples whose length is within [min_range, max_range]
        """
        length_list = []
        for i in range(0, self.num_items):
            length_list.append((i % (self.max_length + 1 - self.min_length)) + self.min_length)

        print(" max is ", max(length_list))
        print(" min is ", min(length_list))
        self.length_list = length_list

    def create_random(self):
        """
        This function creates a list of lengths, the lengths are drawn from a Random distribution.
        Samples are drawn until we have enough samples whose length is within [min_range, max_range]
        """
        length_list = []
        for i in range(self.num_items):
            length_list.append(random.randint(self.min_length, self.max_length))

        print("max is ", max(length_list))
        print("min is ", min(length_list))
        self.length_list = length_list

    def visualize_distribution(self, verbose=True):
        """
        This function visualizes the distribution of the drawn lengths. 
        """
        plt.xlim([0, 65])
        bins = len(set(self.length_list))
        plt.hist(self.length_list, bins = len(set(self.length_list)), alpha=0.5)
        plt.title("Distribution: " + self.distribution)
        plt.xlabel("Length of sample")
        plt.ylabel("count")
        plt.savefig(os.path.join(self.save_dir, "Distribution_visualization.jpg"))
        if verbose:
            plt.show()

    def create_tensors(self):
        """
        This function creates the random numpy arrays, according to the length list.
        It also saves the created arrays to the disk.
        """
        lca_matrices = []
        leaves_list = []

        for mode in ["train", "val"]:
            counter = 0
            first = True
            for idx, len in enumerate(self.length_list):
                lca_matrix = np.random.rand(self.train_events, len, len)
                leaves = np.random.rand(self.train_events, len, self.features_dim)
                lca_matrices = [lca_matrix]
                leaves_list = [leaves]
                self._save_to_drive(lca_matrices, "lcas", mode, counter)
                self._save_to_drive(leaves_list, "leaves", mode, counter)
                counter += 1
                '''
                if idx % self.train_events == 0 and not first:
                    self._save_to_drive(lca_matrices, "lcas", mode, counter)
                    self._save_to_drive(leaves_list, "leaves", mode, counter)
                    lca_matrices = []
                    leaves_list = []
                    counter += 1
                '''
                first = False
            self._save_to_drive(lca_matrices, "lcas", mode, counter)
            self._save_to_drive(leaves_list, "leaves", mode, counter)

    def _save_to_drive(self, tensor_list, identifier, mode, counter, path="./"):
        """
        This function saves a list of tensors to the disk.
        tensor_list: List of tensors to save to disk
        identifier: leaves or lcas
        mode: train or validation
        counter: number identifying the numpy file
        path: directory to save numpy files to
        """
        save_path = self.save_dir if os.path.isdir(self.root_dir) else path
        for tensor in tensor_list:
            np.save(os.path.join(save_path, f"{identifier}_{mode}.{counter}.npy"), tensor)
    
    def save_length_list(self, path="./"):
        """
        This function saves the drawn length list to a file so that it is possible
        to create the numpy arrays or the distribution visualization again just using 
        this file.
        """
        save_path = self.save_dir if os.path.isdir(self.root_dir) else path
        save_path = os.path.join(save_path, "length_list_file")
        with open(save_path, "wb") as fp:
            pickle.dump(self.length_list, fp)
        self.length_list_path = save_path

    def read_length_list(self, path="./"):
        with open(path, "rb") as fp:
            self.length_list = pickle.load(fp)


def main():
    root = "/directory/to/save/numpy/files/in"
    
    #example usage for saving numpy files:
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Gaussian", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Exponential_Right", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Exponential_Left", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Random", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Equal", min_length=2, max_length=60)

    return 0

if __name__ == "__main__":
    main()
