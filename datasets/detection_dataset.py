import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import im_reader

class ImageDetectionDataset(Dataset) :
    def __init__(self, data, mode, label_columns, transform_sequence, grid_size, anchors, image_reader=im_reader, label_reader=None):
        super(ImageDetectionDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data['id'].values.tolist()
        self.image_paths = data['path'].values.tolist()
        self.label_paths = data['label'].values.tolist()
        self.image_reader = image_reader
        self.label_reader = label_reader
        self.dataset_length = len(self.image_paths)
        self.grid_size = grid_size
        self.anchors = anchors
        self.num_anchors = len(anchors)
    def __len__(self) :
        return self.dataset_length
    def get_grid_cell(self,box) :
        cell_size = 1.0/self.grid_size
        x_center, y_center, box_w, box_h, box_ang = box
        i = int(x_center/cell_size)
        j = int(y_center/cell_size)
        x_cell = x_center/cell_size - i
        y_cell = y_center/cell_size - j
        return i,j,x_cell,y_cell,box_w,box_h, box_ang
    def create_target_tensor(self,boxes) :
        target_tensor = np.zeros((self.grid_size,self.grid_size,self.num_anchors,6))
        for box in boxes :
            i,j,x_cell,y_cell,box_w,box_h,box_ang = self.get_grid_cell(box)
            anchor_idx = np.argmin([np.abs(a-box_w)+np.abs(b-box_h)+np.abs(c-box_ang) for a,b,c in self.anchors])
            if target_tensor[j,i,anchor_idx,0] == 0 :
                target_tensor[j,i,anchor_idx,:] = np.array([1,x_cell,y_cell,box_w,box_h, box_ang])
        return target_tensor
    def __getitem__(self,idx) :
        img = self.image_reader(self.image_paths[idx])
        bboxes = self.label_reader(self.label_paths[idx])
        img, bboxes = self.transform_sequence(img, bboxes)
        target = self.create_target_tensor(bboxes)
        return {"inputs":[img],"ground_truths":[target],"debug_info":[self.image_id[idx]]}