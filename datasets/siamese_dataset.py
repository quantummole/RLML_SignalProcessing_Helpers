import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import im_reader, LOAD_MODE


class ImageSiameseDataset(Dataset) :
    def __init__(self, data, mode, classes_per_sample, transform_sequence, reader = im_reader) :
        super(ImageSiameseDataset,self).__init__()
        self.data_groups = data.groupby(["label"])
        self.groups = list(self.data_groups.groups.keys())
        self.group_counts = self.data_groups.count().values
        self.num_groups = len(self.data_groups.groups.keys())
        self.image_paths = data.path.values.tolist()
        self.image_class = data.label.values.tolist()
        self.transform_sequence = transform_sequence
        self.classes_per_sample = classes_per_sample
        self.mode = mode
        self.reader = reader
        self.dataset.length = np.min(self.group_counts)
    def __len__(self) :
        return self.dataset.length
    def __getitem__(self,idx) :
        classes =  np.random.choice(self.groups,size=min([self.classes_per_sample,self.num_groups]),replace=False)
        inputs = []
        labels = []
        for cls in classes :
            paths = np.random.choice(self.data_groups.get_group(cls)["path"].values,size=2,replace=False)
            for path in paths :
                img = self.reader(path)
                if self.transform_sequence :
                    img = self.transform_sequence(img)
                inputs.append(img)
                labels.append(cls)
        return {"inputs":inputs,"ground_truths":labels}