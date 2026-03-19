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
            # 直接通过 index 生成对应文件名
            # 这里假设index中的数字是从0开始的, 如果 index 中的数字表示文件编号减1，直接加1
            # 如果 index 中是直接文件编号，则无需 +1
            self.img_names = [os.path.join(self.data_root, "img_{:04d}.png".format(x + 1)) for x in index]
        else:
            # 如果没有提供 index，则按目录顺序读取所有文件
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

        # 初始化图像路径列表
        self.cam_img_names = []
        self.prj_img_names = []

        # 函数：读取目录内的文件名，并构建一个字典，以文件名为键，存储所在路径
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

        # 读取相机和投影机目录中的文件，分别创建文件名到路径的映射字典
        cam_images = collect_images(cam_dir)
        prj_images = collect_images(prj_dir)

        # 遍历所有文件名，将路径添加到图像路径列表中
        for img_name in cam_images:
            for path in cam_images.get(img_name, []):
                self.cam_img_names.append(path)

        for img_name in prj_images:
            for path in prj_images.get(img_name, []):
                self.prj_img_names.append(path)

        # 最终断言，确保相机图像数量与投影仪图像数量相同
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
                                               index=self.surf_idx)  # multiple surfaces image should better represent surface properties
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
        # 创建一个全零数组，形状为 (height, width, channels, len(index))，其中 height, width, channels 从 im 中获取
        imgs = np.zeros(im.shape + (len(index),), dtype=np.uint8)
        imgs[:, :, :, 0] = im
        for i in range(1, len(index)):
            # imgs的维度为[height,width,channels,len(index)]
            imgs[:, :, :, i] = cv.cvtColor(cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[i] + 1))),
                                           cv.COLOR_BGR2RGB)
        # 将数组 imgs 转换为 PyTorch 张量，并进行维度转换，使通道维度移动到最后,最后将张量中的值除以 255（归一化）并返回
        # 最终返回的张量形状将是(len(index), channels,height, width)
        return torch.Tensor(imgs.transpose(3, 2, 0, 1) / 255)


# class FlowFormerDataset(Dataset):
#     """Dataset adapted for FlowFormer, focusing on camera and projected images."""
#     def readMultipleImages(self, path, index):
#         im = cv.cvtColor(cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[0] + 1))), cv.COLOR_BGR2RGB)
#
#         imgs = np.zeros(im.shape + (len(index),), dtype=np.uint8)
#         imgs[:, :, :, 0] = im
#         for i in range(1, len(index)):
#             imgs[:, :, :, i] = cv.cvtColor(cv.imread(fullfile(path, 'img_{:04d}.png'.format(index[i] + 1))),
#                                            cv.COLOR_BGR2RGB)
#         # return torch.Tensor(imgs.transpose(3, 2, 0, 1) / 255)
#         return torch.from_numpy(imgs.transpose(3, 2, 0, 1)).float() / 255
#
#     def __init__(self, dataset_root, data_list, phase='train', num_train=500, transforms=None, surf_index=None):
#         self.surf_idx = surf_index
#         self.dataset_root = dataset_root
#         self.data_list = data_list
#         self.phase = 'test' if phase == 'valid' else phase
#         self.transforms = transforms
#
#         # Directories for cropped camera images and projector images
#         cam_dir = [fullfile(self.dataset_root, data_name, 'cam', 'crop', self.phase) for data_name in self.data_list]
#         prj_dir = [fullfile(self.dataset_root, self.phase) for data_name in self.data_list]
#         bg_dir = [fullfile(self.dataset_root, data_name, 'cam', 'crop', 'ref') for data_name in self.data_list]
#         # Collecting image file names
#         img_list = sorted(os.listdir(cam_dir[0]))[:num_train]  # Limiting to num_train images
#         self.cam_img_names = [fullfile(dir, img) for img in img_list for dir in cam_dir]
#         self.prj_img_names = [fullfile(dir, img) for img in img_list for dir in prj_dir]
#
#         assert len(self.cam_img_names) == len(self.prj_img_names), 'Mismatch in number of camera and projector images!'
#
#     def __len__(self):
#         return len(self.cam_img_names)
#
#     def __getitem__(self, idx):
#         # Load images
#         cam_img = cv.imread(self.cam_img_names[idx], cv.IMREAD_COLOR)
#         prj_img = cv.imread(self.prj_img_names[idx], cv.IMREAD_COLOR)
#         # 表面背景地址
#         # cam\crop\train\ -> cam\crop\ref
#         surf_img_path = fullfile(self.cam_img_names[idx].split(os.path.sep + self.phase + os.path.sep)[0], 'ref')
#         surf_img_all = self.readMultipleImages(surf_img_path,
#                                                index=self.surf_idx)  # multiple surfaces image should better represent surface properties
#         surf_img_all = surf_img_all.reshape(len(self.surf_idx) * surf_img_all.shape[1], surf_img_all.shape[2],
#                                             surf_img_all.shape[3])
#         # Convert BGR to RGB
#         cam_img = cv.cvtColor(cam_img, cv.COLOR_BGR2RGB)
#         prj_img = cv.cvtColor(prj_img, cv.COLOR_BGR2RGB)
#
#         # Apply transformations
#         # transformation
#         if self.transforms is not None:
#             if self.transforms['cam'] is not None:
#                 cam_img = self.transforms['cam'](cam_img)
#             if self.transforms['prj'] is not None:
#                 prj_img = self.transforms['prj'](prj_img)
#         return cam_img, prj_img, surf_img_all

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

        # 替换路径中的 cam_type（如 crop）为 'ref'，构造 surf 图像路径
        # surf_img_path = self.cam_img_names[idx].replace(os.path.sep + self.cam_type + os.path.sep, os.path.sep + 'ref' + os.path.sep)
        # surf_img_path = os.path.dirname(surf_img_path)  # 去掉文件名
        #
        # surf_img_all = self.readMultipleImages(surf_img_path, index=self.surf_idx)
        # surf_img_all = surf_img_all.reshape(len(self.surf_idx) * surf_img_all.shape[1], surf_img_all.shape[2], surf_img_all.shape[3])

        # cam\crop\train\ -> cam\crop\ref
        surf_img_path = fullfile(self.cam_img_names[idx].split(os.path.sep + self.phase + os.path.sep)[0], 'ref')
        surf_img_all = self.readMultipleImages(surf_img_path,
                                               index=self.surf_idx)  # multiple surfaces image should better represent surface properties
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