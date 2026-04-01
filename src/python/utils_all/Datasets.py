import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2 as cv
from os.path import join as fullfile
import torch.optim as optim


class SimpleDataset(Dataset):
    """Simple dataset."""

    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size
        #
        # # SCNet_surf1_img list
        # img_list = sorted(os.listdir(data_root))
        # if index is not None: img_list = [img_list[x] for x in index]
        #
        # self.img_names = [fullfile(self.data_root, name) for name in img_list]
        if index is not None:
            # Generate the corresponding file names directly from the index
            # Here we assume the numbers in index start from 0.
            # If the numbers in index represent file_number - 1, add 1 directly.
            # If the numbers in index are already the actual file numbers, no +1 is needed.
            self.img_names = [os.path.join(self.data_root, "img_{:04d}.png".format(x + 1)) for x in index]
        else:
            # If no index is provided, read all files in directory order
            img_list = sorted(os.listdir(data_root))
            self.img_names = [os.path.join(self.data_root, name) for name in img_list]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        assert os.path.isfile(img_name), img_name + ' does not exist'
        im = cv.imread(self.img_names[idx])

        # resize image if size is specified
        if self.size is not None:
            im = cv.resize(im, self.size[::-1])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        return im


# CompenNest  dataset
class CompenNetMultiDataset(Dataset):
    """CompenNet Multiple dataset."""

    def __init__(self, dataset_root, data_list, phase='train', data_type='warpSL', surf_idx=[62], transforms=None, CMP=False):
        self.dataset_root = dataset_root
        self.data_list = data_list
        self.phase = phase
        # valid and test use the same prj input images under 'test' folder, the difference is whether the model has seen the projection surface.
        if self.phase == 'valid': self.phase = 'test'
        self.data_type = data_type
        self.surf_idx = surf_idx
        self.transforms = transforms
        self.CMP = CMP

        # image paths
        if not CMP:
            prj_dir = [fullfile(self.dataset_root, self.phase) for data_name in self.data_list]
        else:
            prj_dir = [fullfile(self.dataset_root, data_name, "prj", "cmp_last_iter") for data_name in self.data_list]

        cam_dir = []
        for data_name in self.data_list:
            if CMP:
                cur_path = fullfile(self.dataset_root, self.phase)
            else:
                cur_path = fullfile(self.dataset_root, data_name, 'cam', self.data_type, self.phase)
            if not os.path.isdir(cur_path):
                cur_path = fullfile(self.dataset_root, data_name, 'cam', 'warpSL', self.phase)
            cam_dir.append(cur_path)

        # Initialize image path lists
        self.cam_img_names = []
        self.prj_img_names = []

        # Helper function: read file names in the directory and build a dictionary
        # that uses the file name as the key and stores the corresponding paths
        def collect_images(directories):
            file_dict = {}
            for cur_dir in directories:
                try:
                    img_list = sorted(os.listdir(cur_dir))
                    for img_name in img_list:
                        if img_name not in file_dict:
                            file_dict[img_name] = []
                        file_dict[img_name].append(os.path.join(cur_dir, img_name))
                except FileNotFoundError:
                    print(f"Directory not found: {cur_dir}")
            return file_dict

        # Read files from the camera and projector directories and create mapping dictionaries from file names to paths
        cam_images = collect_images(cam_dir)
        prj_images = collect_images(prj_dir)

        # Iterate over all file names and append the paths to the image path lists
        for img_name in cam_images:
            for path in cam_images.get(img_name, []):
                self.cam_img_names.append(path)

        for img_name in prj_images:
            for path in prj_images.get(img_name, []):
                self.prj_img_names.append(path)

        # Final assertion to ensure the number of camera images matches the number of projector images
        assert len(self.cam_img_names) == len(
            self.prj_img_names), 'Dataset Error: camera image numbers do not match projector image numbers!'

    def __len__(self):
        return len(self.cam_img_names)

    def __getitem__(self, idx):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        # surf
        surf_img_path = fullfile(self.cam_img_names[idx].split(os.path.sep + self.phase + os.path.sep)[0], 'ref')
        # print(f"Surf image path: {self.cam_img_names}")
        # print(f"Surf image path: {surf_img_path}")
        surf_img_all = self.readMultipleImages(surf_img_path,
                                               index=self.surf_idx)  # Multiple surface images should better represent surface properties
        surf_img_all = surf_img_all.reshape(len(self.surf_idx) * surf_img_all.shape[1], surf_img_all.shape[2], surf_img_all.shape[3])

        # cam
        cam_img = cv.cvtColor(cv.imread(self.cam_img_names[idx]), cv.COLOR_BGR2RGB)

        # prj
        prj_img = cv.cvtColor(cv.imread(self.prj_img_names[idx]), cv.COLOR_BGR2RGB)

        # transformation
        if self.transforms is not None:
            if self.transforms['surf'] is not None:
                surf_img_all = self.transforms['surf'](surf_img_all)
            if self.transforms['cam'] is not None:
                cam_img = self.transforms['cam'](cam_img)
            if self.transforms['prj'] is not None:
                prj_img = self.transforms['prj'](prj_img)

        img_sample = {'surf': surf_img_all, 'cam': cam_img, 'prj': prj_img}
        return img_sample

    def readMultipleImages(self, path, index):
        im = cv.cvtColor(cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[0] + 1))), cv.COLOR_BGR2RGB)
        # Create an all-zero array with shape (height, width, channels, len(index)),
        # where height, width, and channels are obtained from im
        imgs = np.zeros(im.shape + (len(index),), dtype=np.uint8)
        imgs[:, :, :, 0] = im
        for i in range(1, len(index)):
            # The dimensions of imgs are [height, width, channels, len(index)]
            imgs[:, :, :, i] = cv.cvtColor(cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[i] + 1))),
                                           cv.COLOR_BGR2RGB)
        # Convert imgs to a PyTorch tensor and permute dimensions so the channel dimension is moved accordingly,
        # then divide the tensor values by 255 for normalization before returning.
        # The returned tensor shape is (len(index), channels, height, width)
        return torch.Tensor(imgs.transpose(3, 2, 0, 1) / 255)

class FlowFormerDataset(Dataset):
    """Dataset adapted for FlowFormer, focusing on camera and projected images."""

    def readMultipleImages(self, path, index):
        im = cv.cvtColor(cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[0] + 1))), cv.COLOR_BGR2RGB)
        imgs = np.zeros(im.shape + (len(index),), dtype=np.uint8)
        imgs[:, :, :, 0] = im
        for i in range(1, len(index)):
            imgs[:, :, :, i] = cv.cvtColor(
                cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[i] + 1))),
                cv.COLOR_BGR2RGB
            )
        return torch.from_numpy(imgs.transpose(3, 2, 0, 1)).float() / 255

    def __init__(self, dataset_root, data_list, phase='train', num_train=500, transforms=None, surf_index=None, cam_type='crop'):
        self.surf_idx = surf_index
        self.dataset_root = dataset_root
        self.data_list = data_list
        self.phase = 'test' if phase == 'valid' else phase
        self.transforms = transforms
        self.cam_type = cam_type  # 'crop' or 'raw'

        cam_dir = [fullfile(self.dataset_root, data_name, 'cam', self.cam_type, self.phase) for data_name in self.data_list]
        prj_dir = [fullfile(self.dataset_root, self.phase) for data_name in self.data_list]

        img_list = sorted(os.listdir(cam_dir[0]))[:num_train]
        self.cam_img_names = [fullfile(dir, img) for img in img_list for dir in cam_dir]
        self.prj_img_names = [fullfile(dir, img) for img in img_list for dir in prj_dir]

        assert len(self.cam_img_names) == len(self.prj_img_names), 'Mismatch in number of camera and projector images!'

    def __len__(self):
        return len(self.cam_img_names)

    def __getitem__(self, idx):
        cam_img = cv.imread(self.cam_img_names[idx], cv.IMREAD_COLOR)
        prj_img = cv.imread(self.prj_img_names[idx], cv.IMREAD_COLOR)

        surf_img_path = fullfile(self.cam_img_names[idx].split(os.path.sep + self.phase + os.path.sep)[0], 'ref')
        surf_img_all = self.readMultipleImages(surf_img_path,
                                               index=self.surf_idx)  # Multiple surface images should better represent surface properties
        surf_img_all = surf_img_all.reshape(len(self.surf_idx) * surf_img_all.shape[1], surf_img_all.shape[2],
                                            surf_img_all.shape[3])
        cam_img = cv.cvtColor(cam_img, cv.COLOR_BGR2RGB)
        prj_img = cv.cvtColor(prj_img, cv.COLOR_BGR2RGB)

        if self.transforms is not None:
            if self.transforms.get('cam') is not None:
                cam_img = self.transforms['cam'](cam_img)
            if self.transforms.get('prj') is not None:
                prj_img = self.transforms['prj'](prj_img)

        return cam_img, prj_img, surf_img_all