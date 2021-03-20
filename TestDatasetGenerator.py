from torch.utils.data import Dataset
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import math
import pickle
import os

class TestDatasetGenerator:
    def __init__(self, num_items=100, distribution="Gaussian", min_length=2, max_length=60, train_events=1, feature_dim=4, root_dir="./"):
        self.num_items = num_items
        self.data = []
        self.labels = []
        self.distribution = distribution if distribution in ["Gaussian", "Exponential_Right", "Exponential_Left", "Random", "Equal"] else "Random"
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
                os.mkdir(os.path.join(self.root_dir, self.distribution))
            self.save_dir = os.path.join(self.root_dir, self.distribution)

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

        assert len(self.length_list) == self.num_items
        self.save_length_list()
        #self.length_list = []
        #self.read_length_list(path="/media/johanna/Volume/Studium/Semester_4/Datamanagement/data/Gaussian_1000/length_list_file")
        #print(self.length_list)

        self.visualize_distribution(verbose=True)
        self.create_tensors()

    def create_gaussian(self):
        center = (int)((self.max_length - self.min_length) / 2) + self.min_length
        scale = math.sqrt(((self.max_length - self.min_length) / 2)) + 4

        # from https://stackoverflow.com/questions/16471763/generating-numbers-with-gaussian-function-in-a-range-using-python
        length_list = []
        for _ in range(self.num_items):
            a = np.random.default_rng().normal(center, scale, 1)
            while a < self.min_length or a > self.max_length + 1:
                a = random.gauss(center, scale)
            length_list.append(int(a))

        print("max is ", max(length_list))
        print("min is ", min(length_list))
        self.length_list = length_list

    def create_exponential_right(self):
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
        length_list = []
        for i in range(0, self.num_items):
            length_list.append((i % (self.max_length + 1 - self.min_length)) + self.min_length)

        print(" max is ", max(length_list))
        print(" min is ", min(length_list))
        self.length_list = length_list

    def create_random(self):
        length_list = []
        for i in range(self.num_items):
            length_list.append(random.randint(self.min_length, self.max_length))

        print("max is ", max(length_list))
        print("min is ", min(length_list))
        self.length_list = length_list

    def visualize_distribution(self, verbose=True):
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
        save_path = self.save_dir if os.path.isdir(self.root_dir) else path
        for tensor in tensor_list:
            np.save(os.path.join(save_path, f"{identifier}_{mode}.{counter}.npy"), tensor)

    def save_length_list(self, path="./"):
        save_path = self.save_dir if os.path.isdir(self.root_dir) else path
        save_path = os.path.join(save_path, "length_list_file")
        with open(save_path, "wb") as fp:
            pickle.dump(self.length_list, fp)
        self.length_list_path = save_path

    def read_length_list(self, path="./"):
        with open(path, "rb") as fp:
            self.length_list = pickle.load(fp)


def main():
    root = "/media/johanna/Volume/Studium/Semester_4/Datamanagement/data/benchmark_10k"
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Gaussian", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Exponential_Right", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Exponential_Left", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Random", min_length=2, max_length=60)
    test = TestDatasetGenerator (num_items=10000, root_dir=root, distribution="Equal", min_length=2, max_length=60)

    return 0

if __name__ == "__main__":
    main()
