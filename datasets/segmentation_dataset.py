import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import im_reader


class ImageSegmentationDataset(Dataset) :
    def __init__(self, data, mode, transform_sequence, image_reader=im_reader, mask_reader=im_reader) :
        super(ImageSegmentationDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data['id'].values.tolist()
        self.image_paths = data['path'].values.tolist()
        self.mask_paths = data['mask'].values.tolist()
        self.image_reader = image_reader
        self.mask_reader = mask_reader
        self.dataset_length = len(self.image_paths)
    def __len__(self) :
        return  self.dataset_length
    def __getitem__(self,idx) :
        impath = self.image_paths[idx]
        mapath = self.mask_paths[idx]
        img = self.image_reader(impath)
        label = self.mask_reader(mapath)
        label = label.astype(np.int64)
        img, gt_img = self.transform_sequence([img,label])
        max_l = np.max(label)
        gt = np.array(gt_img)
        gt = np.clip(gt.round(),0,max_l).astype(np.int64)
        return {"inputs":[img],"ground_truths":[gt],"debug_info":[self.image_id[idx]]}
