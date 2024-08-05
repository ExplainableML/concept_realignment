"""
General utils for training, evaluation and data loading

Adapted from: https://github.com/mateoespinosa/cem/blob/main/cem/data/CUB200/cub_loader.py
AWA2 dataset class taken from: https://github.com/ExplainableML/rdtc/blob/main/utils/data_loader.py
"""

import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from collections import defaultdict
import imageio
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import webdataset as wds
from PIL import Image

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50

DATASET_DIR = os.environ.get("DATASET_DIR", '/mnt/qb/work/bethge/bkr046/DATASETS/Animals_with_Attributes2/')


# Generate a mapping containing all concept groups in CUB generated
# using a simple prefix tree
CONCEPT_GROUP_MAP = None

##########################################################
## ORIGINAL SAMPLER/CLASSES FROM CBM PAPER
##########################################################

class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


##########################################################
## AWA2 Dataset
##########################################################

class AWA2Dataset(Dataset):
    """Animals with Attributes 2 dataset."""
    split_file = 'train_val_test_classification_split.txt'
    data_dir = '' #'awa2'
    # attribute_file = 'predicate-matrix-continuous.txt'
    attribute_file = 'predicate-matrix-binary.txt'

    def __init__(self, root, split, transform=None, return_attributes=True):
        '''
        split = 'train', 'val', or 'test'
        '''
        
        self.root = os.path.join(root, self.data_dir)
        self.split = split
        self.transform = transform
        self.return_attributes = return_attributes

        meta_data = pd.read_csv(os.path.join(self.root,
                                             self.split_file),
                                sep=' ', index_col=0, header=None)
        
        if split == 'train':
            is_train_image = 1
        elif split == 'val':
            is_train_image = 2
        elif split == 'test':
            is_train_image = 0

        self.img_ids = meta_data[meta_data[3] == is_train_image].index.tolist()
        self.id_to_img = meta_data

        raw_mtx = np.loadtxt(os.path.join(self.root,
                                          self.attribute_file))
        # raw_mtx[raw_mtx == -1] = 0
        # raw_mtx = raw_mtx / raw_mtx.max()
        self.attribute_mtx = torch.tensor(raw_mtx, dtype=torch.float)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_meta_data = self.id_to_img[self.id_to_img.index == img_id]
        img_name = img_meta_data.values[0][0]
        img_path = os.path.join(self.root, img_name)

        # img = imageio.imread(img_path, pilmode='RGB')
        # if isinstance(img, np.ndarray):
        #     img = Image.fromarray(img)

        img = Image.open(img_path).convert('RGB')
        
        label = img_meta_data.values[0][1] - 1

        if self.transform:
            img = self.transform(img)

        if self.return_attributes:
            return img, label, self.attribute_mtx[label].flatten()
        return img, label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for
    imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples


def load_data(
    split,
    use_attr,
    no_img,
    batch_size,
    uncertain_label=False,
    n_class_attr=2,
    image_dir='images',
    resampling=False,
    resol=299,
    root_dir='/mnt/qb/work/bethge/bkr046/DATASETS/Animals_with_Attributes2/',
    num_workers=8,
    concept_transform=None,
    label_transform=None,
    path_transform=None,
    is_chexpert=False,
    return_dataset=False,
):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if
    there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change
    sampler.py if necessary
    """
    resized_resol = int(resol * 256/224)
    is_training = (split == 'train')
    if is_training:
        if is_chexpert:
            transform = transforms.Compose([
                transforms.CenterCrop((320, 320)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    else:
        if is_chexpert:
            transform = transforms.Compose([
                transforms.CenterCrop((320, 320)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])

    dataset = AWA2Dataset(
        root=root_dir,
        split=split,
        # transform=transform,
        return_attributes=True)

    if return_dataset:
        return dataset

    if split == 'train':
        # drop_last = True
        drop_last = False
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = StratifiedSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    return loader


def find_class_imbalance(predicates_filename, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    binary_predicates_file_path = os.path.join(DATASET_DIR, predicates_filename)
    binary_predicates = np.loadtxt(binary_predicates_file_path)

    if multiple_attr:
        n_ones = np.sum(binary_predicates, axis=0)
        imbalance = binary_predicates.shape[0]/n_ones - 1

        if attr_idx >= 0:
            return imbalance[attr_idx]
        else:
            return imbalance
    
    else:
        n_ones = np.sum(binary_predicates)
        total = np.size(binary_predicates)
        imbalance = total/n_ones - 1
        
        return imbalance


##########################################################
## SIMPLIFIED LOADER FUNCTION FOR STANDARDIZATION
##########################################################


# def generate_data(
#     config,
#     root_dir=DATASET_DIR,
#     seed=42,
#     output_dataset_vars=False,
#     rerun=False,
#     return_dataset=False,
# ):
#     if root_dir is None:
#         root_dir = DATASET_DIR
#     # base_dir = os.path.join(root_dir, 'class_attr_data_10')
#     seed_everything(seed)
#     # train_data_path = os.path.join(base_dir, 'train.pkl')
#     predicates_filename = 'predicate-matrix-binary.txt'
#     if config.get('weight_loss', False):
#         imbalance = find_class_imbalance(predicates_filename, True)
#     else:
#         imbalance = None
#
#     # val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
#     # test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
#     sampling_percent = config.get("sampling_percent", 1)
#     sampling_groups = config.get("sampling_groups", False)
#
#     if CONCEPT_GROUP_MAP is not None:
#         concept_group_map = CONCEPT_GROUP_MAP.copy()
#     else:
#         concept_group_map = None
#
#     class2concepts = np.loadtxt(os.path.join(root_dir, predicates_filename))
#
#     n_concepts = class2concepts.shape[1]
#
#     if sampling_percent != 1:
#         raise Exception("Sampling not implemented")
#         # Do the subsampling
#         if sampling_groups:
#             new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
#             selected_groups_file = os.path.join(
#                 DATASET_DIR,
#                 f"selected_groups_sampling_{sampling_percent}.npy",
#             )
#             if (not rerun) and os.path.exists(selected_groups_file):
#                 selected_groups = np.load(selected_groups_file)
#             else:
#                 selected_groups = sorted(
#                     np.random.permutation(len(concept_group_map))[:new_n_groups]
#                 )
#                 np.save(selected_groups_file, selected_groups)
#             selected_concepts = []
#             group_concepts = [x[1] for x in concept_group_map.items()]
#             for group_idx in selected_groups:
#                 selected_concepts.extend(group_concepts[group_idx])
#             selected_concepts = sorted(set(selected_concepts))
#         else:
#             new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
#             selected_concepts_file = os.path.join(
#                 DATASET_DIR,
#                 f"selected_concepts_sampling_{sampling_percent}.npy",
#             )
#             if (not rerun) and os.path.exists(selected_concepts_file):
#                 selected_concepts = np.load(selected_concepts_file)
#             else:
#                 selected_concepts = sorted(
#                     np.random.permutation(n_concepts)[:new_n_concepts]
#                 )
#                 np.save(selected_concepts_file, selected_concepts)
#         # Then we also have to update the concept group map so that
#         # selected concepts that were previously in the same concept
#         # group are maintained in the same concept group
#         new_concept_group = {}
#         remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
#         selected_concepts_set = set(selected_concepts)
#         for selected_concept in selected_concepts:
#             for concept_group_name, group_concepts in concept_group_map.items():
#                 if selected_concept in group_concepts:
#                     if concept_group_name in new_concept_group:
#                         # Then we have already added this group
#                         continue
#                     # Then time to add this group!
#                     new_concept_group[concept_group_name] = []
#                     for other_concept in group_concepts:
#                         if other_concept in selected_concepts_set:
#                             # Add the remapped version of this concept
#                             # into the concept group
#                             new_concept_group[concept_group_name].append(
#                                 remap[other_concept]
#                             )
#         # And update the concept group map accordingly
#         concept_group_map = new_concept_group
#         print("\t\tSelected concepts:", selected_concepts)
#         print(f"\t\tUpdated concept group map (with {len(concept_group_map)} groups):")
#         for k, v in concept_group_map.items():
#             print(f"\t\t\t{k} -> {v}")
#
#         def concept_transform(sample):
#             if isinstance(sample, list):
#                 sample = np.array(sample)
#             return sample[selected_concepts]
#
#         # And correct the weight imbalance
#         if config.get('weight_loss', False):
#             imbalance = np.array(imbalance)[selected_concepts]
#         n_concepts = len(selected_concepts)
#     else:
#         concept_transform = None
#
#
#     train_dl = load_data(
#         split='train',
#         # pkl_paths=[train_data_path],
#         use_attr=True,
#         no_img=False,
#         batch_size=config['batch_size'],
#         uncertain_label=False,
#         n_class_attr=2,
#         image_dir='images',
#         resampling=False,
#         root_dir=root_dir,
#         num_workers=config['num_workers'],
#         concept_transform=concept_transform,
#         return_dataset=return_dataset
#     )
#
#     val_dl = load_data(
#         split='val',
#         # pkl_paths=[val_data_path],
#         use_attr=True,
#         no_img=False,
#         batch_size=config['batch_size'],
#         uncertain_label=False,
#         n_class_attr=2,
#         image_dir='images',
#         resampling=False,
#         root_dir=root_dir,
#         num_workers=config['num_workers'],
#         concept_transform=concept_transform,
#         return_dataset=return_dataset
#     )
#
#     test_dl = load_data(
#         split='test',
#         # pkl_paths=[test_data_path],
#         use_attr=True,
#         no_img=False,
#         batch_size=config['batch_size'],
#         uncertain_label=False,
#         n_class_attr=2,
#         image_dir='images',
#         resampling=False,
#         root_dir=root_dir,
#         num_workers=config['num_workers'],
#         concept_transform=concept_transform,
#         return_dataset=return_dataset
#     )
#     if not output_dataset_vars:
#         return train_dl, val_dl, test_dl, imbalance
#     return (
#         train_dl,
#         val_dl,
#         test_dl,
#         imbalance,
#         (n_concepts, N_CLASSES, concept_group_map),
#     )

def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    return_dataset=False,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    # base_dir = os.path.join(root_dir, 'class_attr_data_10')
    seed_everything(seed)
    # train_data_path = os.path.join(base_dir, 'train.pkl')
    predicates_filename = 'predicate-matrix-binary.txt'
    if config.get('weight_loss', False):
        imbalance = find_class_imbalance(predicates_filename, True)
    else:
        imbalance = None

    # val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    # test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)

    if CONCEPT_GROUP_MAP is not None:
        concept_group_map = CONCEPT_GROUP_MAP.copy()
    else:
        concept_group_map = None

    class2concepts = np.loadtxt(os.path.join(root_dir, predicates_filename))

    n_concepts = class2concepts.shape[1]

    if sampling_percent != 1:
        raise Exception("Sampling not implemented")
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                DATASET_DIR,
                f"selected_groups_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_groups_file):
                selected_groups = np.load(selected_groups_file)
            else:
                selected_groups = sorted(
                    np.random.permutation(len(concept_group_map))[:new_n_groups]
                )
                np.save(selected_groups_file, selected_groups)
            selected_concepts = []
            group_concepts = [x[1] for x in concept_group_map.items()]
            for group_idx in selected_groups:
                selected_concepts.extend(group_concepts[group_idx])
            selected_concepts = sorted(set(selected_concepts))
        else:
            new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                DATASET_DIR,
                f"selected_concepts_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(n_concepts)[:new_n_concepts]
                )
                np.save(selected_concepts_file, selected_concepts)
        # Then we also have to update the concept group map so that
        # selected concepts that were previously in the same concept
        # group are maintained in the same concept group
        new_concept_group = {}
        remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
        selected_concepts_set = set(selected_concepts)
        for selected_concept in selected_concepts:
            for concept_group_name, group_concepts in concept_group_map.items():
                if selected_concept in group_concepts:
                    if concept_group_name in new_concept_group:
                        # Then we have already added this group
                        continue
                    # Then time to add this group!
                    new_concept_group[concept_group_name] = []
                    for other_concept in group_concepts:
                        if other_concept in selected_concepts_set:
                            # Add the remapped version of this concept
                            # into the concept group
                            new_concept_group[concept_group_name].append(
                                remap[other_concept]
                            )
        # And update the concept group map accordingly
        concept_group_map = new_concept_group
        print("\t\tSelected concepts:", selected_concepts)
        print(f"\t\tUpdated concept group map (with {len(concept_group_map)} groups):")
        for k, v in concept_group_map.items():
            print(f"\t\t\t{k} -> {v}")

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        # And correct the weight imbalance
        if config.get('weight_loss', False):
            imbalance = np.array(imbalance)[selected_concepts]
        n_concepts = len(selected_concepts)
    else:
        concept_transform = None

    # transforms
    resol = 299

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
        transforms.RandomResizedCrop(resol),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # implicitly divides by 255
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(resol),
        transforms.ToTensor(),  # implicitly divides by 255
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    # load attributes
    attribute_file = '/mnt/qb/work/bethge/bkr046/DATASETS/Animals_with_Attributes2/predicate-matrix-binary.txt'
    raw_mtx = np.loadtxt(attribute_file)
    attribute_mtx = torch.tensor(raw_mtx, dtype=torch.float)

    # Define a simple function to decode the samples and apply transformations
    def decoder(sample, transform):
        # key = sample["__key__"]
        image = pickle.loads(sample["input.pyd"])  # deserialize the image
        label = pickle.loads(sample["output.pyd"])  # deserialize the label

        # Convert the image to PIL format if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Apply the transformations
        image = transform(image)

        attributes = attribute_mtx[label].flatten()

        return image, label, attributes


    DATA_SAVE_PATH = "/mnt/qb/work/bethge/bkr046/DATASETS/Animals_with_Attributes2/webdataset/"
    split_file = '/mnt/qb/work/bethge/bkr046/DATASETS/Animals_with_Attributes2/train_val_test_classification_split.txt'
    meta_data = pd.read_csv(split_file, sep=' ', index_col=0, header=None)

    # load training data
    train_dataset = wds.WebDataset(DATA_SAVE_PATH + "train_dataset.tar")
    train_decoder = lambda sample: decoder(sample, train_transform)
    train_dataset = train_dataset.map(train_decoder)
    len_train_dataset = len(meta_data[meta_data[3] == 1].index.tolist())
    train_dataset = train_dataset.with_length(len_train_dataset)
    train_dataset = train_dataset.shuffle(len_train_dataset, initial=len_train_dataset)
    train_dl = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

    # load val data
    val_dataset = wds.WebDataset(DATA_SAVE_PATH + "val_dataset.tar")
    val_decoder = lambda sample: decoder(sample, test_transform)
    val_dataset = val_dataset.map(val_decoder)
    len_val_dataset = len(meta_data[meta_data[3] == 2].index.tolist())
    val_dataset = val_dataset.with_length(len_val_dataset)
    val_dl = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

    # load test data
    test_dataset = wds.WebDataset(DATA_SAVE_PATH + "test_dataset.tar")
    test_decoder = lambda sample: decoder(sample, test_transform)
    test_dataset = test_dataset.map(test_decoder)
    len_test_dataset = len(meta_data[meta_data[3] == 0].index.tolist())
    test_dataset = test_dataset.with_length(len_test_dataset)
    test_dl = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, N_CLASSES, concept_group_map),
    )



