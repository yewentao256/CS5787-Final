from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from os.path import join, splitext, basename
from glob import glob

CORP_REGION_SIZE = 80

class dataset_norm(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root="", transforms=None, imgSize=192, inputsize=128):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.transforms = transforms
        self.imgSize = imgSize
        self.inputsize = inputsize
        self.img_list = sorted(glob(join(root, "*.jpg")))
        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        img = Image.open(self.img_list[index])
        i = (self.imgSize - self.inputsize) // 2

        img = self.transforms(img)

        iner_img = img[:, i : i + self.inputsize, :]
        iner_img = iner_img[:, :, i : i + self.inputsize]
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:, i : i + self.inputsize, i : i + self.inputsize] = iner_img

        return img, mask_img

    def __len__(self):
        return self.size


class dataset_two_corps(Dataset):

    def __init__(self, root="", transforms=None, imgSize=192, inputsize=CORP_REGION_SIZE):
        self.transforms = transforms
        self.imgSize = imgSize
        self.inputsize = inputsize
        self.img_list = sorted(glob(join(root, "*.jpg")))
        self.size = len(self.img_list)

    def __getitem__(self, index):
        index = index % self.size
        img = Image.open(self.img_list[index])
        img = self.transforms(img)

        # left down
        i1 = self.imgSize - self.inputsize
        j1 = 0
        # right up
        i2 = 0
        j2 = self.imgSize - self.inputsize
        left_bottom = img[:, i1:i1 + self.inputsize, j1:j1 + self.inputsize]
        right_top = img[:, i2:i2 + self.inputsize, j2:j2 + self.inputsize]

        mask_img = np.zeros((3, self.imgSize, self.imgSize))
        mask_img[:, i1:i1 + self.inputsize, j1:j1 + self.inputsize] = left_bottom.numpy()
        mask_img[:, i2:i2 + self.inputsize, j2:j2 + self.inputsize] = right_top.numpy()
        
        return img, mask_img

    def __len__(self):
        return self.size
    

class dataset_arbi(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(
        self, root="", transforms=None, imgSize=192, inputsize=128, pred_step=1
    ):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize

        file_list = sorted(glob(join(root, "*.png")))

        for name in file_list:
            img = Image.open(name)
            # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
            if (img.size[0] >= 0) and (img.size[1] >= 0):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert("RGB")
        i = (self.imgSize - self.inputsize) // 2
        j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        # iner_img = img[:, i:i + self.inputsize, :]
        # iner_img = iner_img[:, :, i:i + self.inputsize]
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
            # mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i : i + self.imgSize, i : i + self.imgSize] = img
        else:
            mask_img[:, i : i + self.inputsize, i : i + self.inputsize] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size
