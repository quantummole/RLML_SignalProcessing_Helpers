import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import im_reader

class ImageJigSawDataset(Dataset) :
    def __init__(self, data, mode, jigsaw_size, transform_sequence, reader=im_reader):
        super(ImageJigSawDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data['id'].values.tolist()
        self.image_paths = data['path'].values.tolist()
        self.reader = reader
        self.jigsaw_size = jigsaw_size
        self.dataset_length = len(self.image_paths)
    def __len__(self) :
        return self.dataset_length
    def get_shuffled_image(self, im):
        permutation = np.random.permutation(self.jigsaw_size*self.jigsaw_size)
        jh, jw = self.jigsaw_size
        m, n, c = im.shape
        assert m%jh == 0 and n%jw == 0, "Image size not divisible by jigsaw size"
        gh, gw = m//jh, n//jw
        shuffled_image = im.reshape((self.jigsaw_size,gh,self.jigsaw_size,gw,c))
        shuffled_image = shuffled_image.transpose((0,2,1,3,4))
        shuffled_image = shuffled_image.reshape((self.jigsaw_size*self.jigsaw_size,gh,gw,c))
        shuffled_image = shuffled_image[permutation].reshape((self.jigsaw_size,self.jigsaw_size,gh,gw, c))
        shuffled_image = shuffled_image.transpose((0,2,1,3,4)).reshape((m,n,c))
        return shuffled_image
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        img = self.reader(path)
        img = self.transform_sequence(img)
        shuffled_img = self.get_shuffled_image(img)
        return {"inputs":[img],"ground_truths":[shuffled_img],"debug_info":[('jigsaw',self.image_id[idx])]}