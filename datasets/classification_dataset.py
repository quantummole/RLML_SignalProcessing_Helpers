import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import im_reader

class ImageClassificationDataset(Dataset) :
    def __init__(self, data, mode, label_columns, transform_sequence, reader=im_reader):
        super(ImageClassificationDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data['id'].values.tolist()
        self.image_paths = data['path'].values.tolist()
        self.image_class = data.ix[:,label_columns].values.tolist()
        self.reader = reader
        self.dataset_length = len(self.image_paths)
    def __len__(self) :
        return self.dataset_length
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        img = self.reader(path)
        img = self.transform_sequence(img)
        im = np.array(img)
        label = self.image_class[idx].astype(np.long)
        return {"inputs":[im],"ground_truths":[label],"debug_info":[self.image_id[idx]]}
