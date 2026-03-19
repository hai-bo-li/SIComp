
# import sys
# # sys.path.insert(0, r"H:/home/lihaibo/Tutor_file")
# from pathlib import Path
# # Add the parent directory (SIComp/src/python) to sys.path
# sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import os
import sys
from os.path import join as fullfile
import numpy as np
import cv2 as cv
import math
import random
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import DataParallel
from . import pytorch_ssim
# from pytorch_ssim import *
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
import torch.nn.functional as F
import inspect
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torchvision import transforms as cv_transforms
from openpyxl import Workbook, load_workbook
# from Datasets import SimpleDataset
# from Datasets import SimpleDataset
from . import Datasets
# from FF_CompenUltra_lab.src.python import Datasets
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from typing import List, Optional


def make_colorwheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col += YG
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col += CB
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def save_flow_to_vismap(flow_tensor, save_path, max_flow=None):
    """
    Enhanced visualization mode:
    - Moderate saturation: controlled around 180 for vivid but readable colors
    - Strong contrast: bright background with clear deformation regions
    """
    # 1. Format conversion [1, 2, H, W] -> [H, W, 2]
    flow_uv = flow_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    u, v = flow_uv[:, :, 0], flow_uv[:, :, 1]

    # 2. Compute polar coordinates
    mag, ang = cv.cartToPolar(u, v)

    # 3. Determine the displacement reference upper bound
    if max_flow is None:
        # Use the 98th percentile to keep the overall tone stable
        reference_max = np.percentile(mag, 98) + 1e-5
    else:
        reference_max = max_flow

    h, w = mag.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # [H - Hue] map direction to (0-179)
    hsv[..., 0] = ang * 180 / np.pi / 2

    # [S - Saturation] core adjustment: dynamic range 30-180
    # Linearly map to 0-1
    sat_normalized = np.clip(mag / reference_max, 0, 1)
    # Map to the range from 30 (static) to 180 (maximum displacement)
    # Use power(0.9) to slightly boost the visibility of low-to-mid motion
    sat = (np.power(sat_normalized, 0.9) * 150) + 30
    hsv[..., 1] = sat.astype(np.uint8)

    # [V - 亮度] 保持在 250，明亮清晰
    hsv[..., 2] = 250

    # 5. 转换并保存 (使用 BGR 适配 cv2.imwrite)
    vis_image = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv.imwrite(save_path, vis_image)

def get_linux_style_dataset_list(
    dataset_root: str,
    exclude_prefixes: Optional[List[str]] = None,
    include_prefixes: Optional[List[str]] = None
) -> List[str]:
    """
    返回 setups 子目录中所有满足条件的路径，格式为 Linux 风格（使用 '/' 分隔符）。

    Args:
        dataset_root (str): 根目录路径，如 'H:/Valid_datasets/CompenHR_datasets'
        exclude_prefixes (List[str], optional): 排除这些前缀的目录名
        include_prefixes (List[str], optional): 只包含这些前缀的目录名（优先级高于 exclude）

    Returns:
        List[str]: 形如 ['setups/Block1', 'setups/Fruits_Vegetables2', ...] 的路径列表
    """
    exclude_prefixes = exclude_prefixes or []
    setups_dir = Path(dataset_root) / 'setups'

    if not setups_dir.exists():
        print(f"[Warning] 目录不存在: {setups_dir}")
        return []

    all_dirs = [name for name in os.listdir(setups_dir) if (setups_dir / name).is_dir()]

    if include_prefixes:
        filtered_dirs = [
            name for name in all_dirs if any(name.startswith(p) for p in include_prefixes)
        ]
    else:
        filtered_dirs = [
            name for name in all_dirs if not any(name.startswith(p) for p in exclude_prefixes)
        ]

    data_list = [os.path.join("setups", name).replace("\\", "/") for name in filtered_dirs]

    # ✅ 打印输出
    print("数据列表如下：")
    for path in data_list:
        print(f"  {path}")
    print(f"\n总计：{len(data_list)} 个 setup 目录")

    return data_list

def visualize_warp(original, warped, mask, warped_clean, tag='debug', save_path='./debug'):
    os.makedirs(save_path, exist_ok=True)
    b = original.size(0)
    for i in range(b):
        orig_img = to_pil_image(original[i].cpu())
        warped_img = to_pil_image(warped[i].cpu())
        mask_img = to_pil_image(mask[i].expand(3, -1, -1).cpu())  # 单通道变成3通道
        warped_clean_img = to_pil_image(warped_clean[i].cpu())

        orig_img.save(os.path.join(save_path, f'{tag}_b{i}_orig.png'))
        warped_img.save(os.path.join(save_path, f'{tag}_b{i}_warped.png'))
        mask_img.save(os.path.join(save_path, f'{tag}_b{i}_mask.png'))
        warped_clean_img.save(os.path.join(save_path, f'{tag}_b{i}_warpedclean.png'))

def load_model_with_bias_resize(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        if "relative_position_bias_table" in k:
            pretrained_bias = state_dict[k]
            current_bias = model.state_dict()[k]
            if pretrained_bias.shape != current_bias.shape:
                print(f"↔️ 自动插值: {k} from {pretrained_bias.shape} to {current_bias.shape}")
                # 旧的是 [L, num_heads]
                num_heads = current_bias.shape[1]
                old_len = int(pretrained_bias.shape[0] ** 0.5)
                new_len = int(current_bias.shape[0] ** 0.5)
                pretrained_bias = pretrained_bias.reshape(1, num_heads, old_len, old_len)
                interpolated = torch.nn.functional.interpolate(
                    pretrained_bias,
                    size=(new_len, new_len),
                    mode="bicubic",
                    align_corners=False,
                )
                state_dict[k] = interpolated.reshape(current_bias.shape)

    model.load_state_dict(state_dict, strict=False)

data_transforms = {'surf': None,  # surf不用转化是因为在dataset里面已经处理成pytorch的训练格式了
                   'cam': cv_transforms.Compose([cv_transforms.ToTensor()]),
                   'prj': cv_transforms.Compose([cv_transforms.ToTensor()])}


def worker_init_fn(worker_id, cfg):
    worker_seed = cfg.trainer.randseed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# def create_dataloader(train_datasets, test_dataset, cfg):
#     # 创建数据生成器
#     data_gen = torch.Generator()
#     data_gen.manual_seed(cfg.trainer.randseed)
#
#     # --- 关键修改：使用 partial 包装函数 ---
#     # 这会创建一个新函数，自动把 cfg 传给你的 worker_init_fn
#     wrapped_init_fn = functools.partial(worker_init_fn, cfg=cfg)
#
#     # 合并训练数据集
#     Concat_dataset = ConcatDataset(train_datasets)
#     Concat_dataloader = DataLoader(
#         Concat_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers,
#         generator=data_gen,
#         worker_init_fn=wrapped_init_fn  # 使用包装后的函数
#     )
#
#     # 创建验证数据集
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers,
#         generator=data_gen,
#         worker_init_fn=wrapped_init_fn  # 使用包装后的函数
#     )
#
#     return Concat_dataloader, test_loader
#linux上面不用序列化
def create_dataloader(train_datasets, test_dataset, cfg):
    # 创建数据生成器
    data_gen = torch.Generator()
    data_gen.manual_seed(cfg.trainer.randseed)

    # 合并训练数据集
    Concat_dataset = ConcatDataset(train_datasets)
    Concat_dataloader = DataLoader(
        Concat_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=data_gen,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, cfg)  # 传递 cfg 给 worker_init_fn
    )

    # 创建验证数据集
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        generator=data_gen,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, cfg)  # 同样传递 cfg 给 worker_init_fn
    )

    return Concat_dataloader, test_loader


def getLargestRect(imBW, aspectRatio):
    h, w = imBW.shape

    # find candidate rect centers using distance transform
    imDist = cv.distanceTransform(imBW, cv.DIST_L2, 3)
    cy, cx = np.where(imDist > np.percentile(imDist[imBW == 0], 90))

    # only use a subset for speedup
    np.random.seed(1)
    idx = np.random.choice(len(cx), min(200, len(cx)), replace=False)
    cx = cx[idx]
    cy = cy[idx]

    maxH = 0

    for i in range(len(cy)):
        lx, rx, ty, by = cx[i], cx[i], cy[i], cy[i]

        while True:
            curH = by - ty + 1
            dx = round(curH * aspectRatio) - round((curH - 1) * aspectRatio)

            if ty - 1 >= 0 and lx - dx >= 0:
                imRect = imBW[ty - 1:by + 1, lx - dx:rx + 1]
                if np.count_nonzero(imRect == 0) == 0:
                    ty -= 1
                    lx -= dx
                else:
                    break
            else:
                break

            curH = by - ty + 1
            dx = round(curH * aspectRatio) - round((curH - 1) * aspectRatio)

            if by + 1 < h and rx + dx < w:
                imRect = imBW[ty:by + 2, lx:rx + dx + 2]
                if np.count_nonzero(imRect == 0) == 0:
                    by += 1
                    rx += dx
                else:
                    break
            else:
                break

        curH = by - ty + 1
        if curH > maxH:
            maxH = curH
            bestx, besty = cx[i], cy[i]
            bestRect = [lx, ty, rx - lx + 1, by - ty + 1]
        elif curH == maxH:
            bestx = np.append(bestx, cx[i])
            besty = np.append(besty, cy[i])
            bestRect = np.vstack((bestRect, [lx, ty, rx - lx + 1, by - ty + 1]))

    idx = np.argmax(imDist[besty, bestx])
    rect = bestRect[idx]

    return rect


# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Get_dataLoader(img_dir, size=None, index=None, cfg=None):
    img_dataset = Datasets.SimpleDataset(img_dir, index=index, size=size)
    data_loader = DataLoader(img_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)
    return data_loader


# Same as np.repeat, while torch.repeat works as np.tile
def repeat_np(a, repeats, dim):
    '''
    Substitute for numpy's repeat function. Source from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    '''

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)]))
    else:
        order_index = torch.LongTensor(
            torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)


# save 4D np.ndarray or torch tensor to image files
def saveImgs(inputData, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if type(inputData) is torch.Tensor:
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)

    else:
        imgs = inputData

    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs[:, :, :, ::-1] * 255)  # convert to BGR and uint8 for opencv
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(i + 1)
        cv.imwrite(fullfile(dir, file_name), imgs[i, :, :, :])  # faster than PIL or scipy


# compute PSNR
def psnr(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return 10 * math.log10(1 / l2_fun(x, y))


# compute RMSE
def rmse(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        l2_fun = nn.MSELoss()
        return math.sqrt(l2_fun(x, y).item() * 3)  # only works for RGB, for grayscale, don't multiply by 3


# compute ssim
def ssim(x, y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device) if x.device.type != device.type else x
    y = y.to(device) if y.device.type != device.type else y

    with torch.no_grad():
        return pytorch_ssim.ssim(x, y).item()


# compute psnr, rmse and ssim
def computeMetrics(x, y) -> object:
    l2_fun = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    last_loc = 0
    metric_mse, metric_ssim = 0., 0.

    num_imgs = x.shape[0]
    batch_size = 50 if num_imgs > 50 else num_imgs  # Make sure mod(num_imgs, batch_size) == 0

    with torch.no_grad():
        for i in range(0, num_imgs // batch_size):
            idx = range(last_loc, last_loc + batch_size)
            x_batch = x[idx, :, :, :].to(device) if x.device.type != 'cuda' else x[idx, :, :, :]
            y_batch = y[idx, :, :, :].to(device) if y.device.type != 'cuda' else y[idx, :, :, :]

            # compute mse and ssim
            metric_mse += l2_fun(x_batch, y_batch).item() * batch_size
            metric_ssim += ssim(x_batch, y_batch) * batch_size

            last_loc += batch_size

        # average
        metric_mse /= num_imgs
        metric_ssim /= num_imgs

        # rmse and psnr
        metric_rmse = math.sqrt(metric_mse * 3)  # 3 channel image
        metric_psnr = 10 * math.log10(1 / metric_mse)

    return metric_psnr, metric_rmse, metric_ssim


# count the number of parameters of a model
def countParameters(model):
    return sum([param.numel() for param in model.parameters() if param.requires_grad])


# generate training title string
def optionToString(train_option):
    return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(train_option['data_name'], train_option['model_name'],
                                                  train_option['loss'],
                                                  train_option['num_train'], train_option['batch_size'],
                                                  train_option['max_iters'],
                                                  train_option['lr'], train_option['lr_drop_ratio'],
                                                  train_option['lr_drop_rate'],
                                                  train_option['l2_reg'])


def log_surrogate(log_dir, current_time, data_name, model_name, loss_function, num_train,
                  batch_size, valid_psnr, valid_rmse, valid_ssim, valid_diff, valid_lpips):
    """
    记录训练结果到日志文件（标题只在文件首次创建时写入）

    参数:
        log_dir:        日志目录路径
        current_time:   外部传入的时间字符串
        ... (其他参数同前)
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, f"{current_time}_surrogate.txt")

    # 检查文件是否已存在（决定是否需要写标题）
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a') as log_file:
        if write_header:
            title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
            log_file.write(title_str.format(
                'data_name', 'model_name', 'loss_function',
                'num_train', 'batch_size',
                'valid_psnr', 'valid_rmse', 'valid_ssim', "valid_diff", "valid_lpips"
            ))

        ret_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
        log_file.write(ret_str.format(
            data_name, model_name, loss_function,
            str(num_train), str(batch_size),
            valid_psnr, valid_rmse, valid_ssim, valid_diff, valid_lpips
        ))

    print(f"Results appended to: {log_path}")


def save_test_log_one_line(folder_path, model_name, batch_size, lr, loss, uncmp_psnr, uncmp_rmse,
                           uncmp_ssim, uncmp_deltaE, uncmp_lpips, psnr, rmse, ssim, deltaE,
                           lpips, txt_file_name="test_log.txt", excel_file_name="test_log.xlsx"):
    """
    同时以文本文件和Excel文件格式保存训练日志到指定的文件夹路径，每栏一列，并在每次调用时追加新行。
    学习率 (lr) 保留小数点后7位。

    :param folder_path: 日志文件夹路径。
    :param model_name: 模型名称。
    :param batch_size: 批大小。
    :param lr: 学习率 (Learning Rate)。
    :param loss: 当前的 loss 描述字符串。
    :param uncmp_psnr: 原始 PSNR 指标。
    :param uncmp_rmse: 原始 RMSE 指标。
    :param uncmp_ssim: 原始 SSIM 指标。
    :param uncmp_deltaE: 原始 DeltaE 指标。
    :param uncmp_lpips: 原始 LPIPS 指标。
    :param psnr: 当前 PSNR 值。
    :param rmse: 当前 RMSE 值。
    :param ssim: 当前 SSIM 值。
    :param deltaE: 当前 DeltaE 值。
    :param lpips: 当前 LPIPS 值。
    :param txt_file_name: 文本日志文件名（默认为 "test_log.txt"）。
    :param excel_file_name: Excel日志文件名（默认为 "test_log.xlsx"）。
    """
    try:
        # 拼接日志文件的完整路径
        txt_log_path = os.path.join(folder_path, txt_file_name)
        excel_log_path = os.path.join(folder_path, excel_file_name)

        # 确保文件夹存在（如果不存在，就递归创建）
        os.makedirs(folder_path, exist_ok=True)

        # 定义标题栏（字段名称）和每列的宽度
        title = [
            "Time", "Model Name", "Batch_Size", "Learning_Rate", "Loss",
            "UnCmp_PSNR", "UnCmp_RMSE", "UnCmp_SSIM",
            "UnCmp_DeltaE", "UnCmp_LPIPS",
            "PSNR", "RMSE", "SSIM", "DeltaE", "LPIPS"
        ]
        widths = [20, 25, 15, 20, 20, 20, 20, 20, 20, 20, 12, 12, 12, 12, 12]

        # ===== 保存到文本文件 =====
        # 检查文本文件是否存在
        if not os.path.exists(txt_log_path):
            with open(txt_log_path, "w", encoding="utf-8") as txt_file:
                # 将标题格式化为固定的列宽并居中对齐
                title_line = "".join([col.center(width) for col, width in zip(title, widths)]) + "\n"
                txt_file.write(title_line)
                # print(f"Log TXT file created with headers: {txt_log_path}")

        # 获取当前时间字符串
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将日志内容格式化为固定的列宽并居中对齐，学习率保留7位小数
        log_message = "".join([
            str(current_time).center(widths[0]),
            str(model_name).center(widths[1]),
            str(batch_size).center(widths[2]),
            f"{lr:.7f}".center(widths[3]),  # 修改处：保留7位小数
            str(loss).center(widths[4]),
            f"{uncmp_psnr:.4f}".center(widths[5]),
            f"{uncmp_rmse:.4f}".center(widths[6]),
            f"{uncmp_ssim:.4f}".center(widths[7]),
            f"{uncmp_deltaE:.4f}".center(widths[8]),
            f"{uncmp_lpips:.4f}".center(widths[9]),
            f"{psnr:.4f}".center(widths[10]),
            f"{rmse:.4f}".center(widths[11]),
            f"{ssim:.4f}".center(widths[12]),
            f"{deltaE:.4f}".center(widths[13]),
            f"{lpips:.4f}".center(widths[14]) + "\n"
        ])

        # 追加模式写入日志内容到文本文件
        with open(txt_log_path, "a", encoding="utf-8") as txt_file:
            txt_file.write(log_message)

        # print(f"Log entry appended to TXT file: {txt_log_path}")

        # ===== 保存到Excel文件 =====
        # 检查Excel文件是否存在
        if not os.path.exists(excel_log_path):
            # 创建一个新的Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Test_Log"  # 可选：设置工作表名称
            ws.append(title)  # 写入标题
            wb.save(excel_log_path)
            # print(f"Log Excel file created with headers: {excel_log_path}")

        # 打开Workbook
        wb = load_workbook(excel_log_path)
        ws = wb.active

        # 创建日志数据行，学习率保留7位小数
        log_row = [
            current_time,
            model_name,
            batch_size,
            round(lr, 7),  # 修改处：保留7位小数
            loss,
            round(uncmp_psnr, 4),
            round(uncmp_rmse, 4),
            round(uncmp_ssim, 4),
            round(uncmp_deltaE, 4),
            round(uncmp_lpips, 4),
            round(psnr, 4),
            round(rmse, 4),
            round(ssim, 4),
            round(deltaE, 4),
            round(lpips, 4)
        ]

        # Append the new row
        ws.append(log_row)

        # 保存Workbook
        wb.save(excel_log_path)

        # print(f"Log entry appended to Excel file: {excel_log_path}")

    except Exception as e:
        print(f"Error saving log files: {e}")


def save_vis_line(vis, win, save_path):
    """
    保存 Visdom 图表为图片，并在曲线中只显示最后一个数据点的数值，第0轮的点不连接到原点。

    :param vis: Visdom 实例
    :param win: Visdom 图表窗口名称
    :param save_path: 保存图片的路径
    """
    # 获取窗口的数据
    window_data = vis.get_window_data(win)
    if not window_data:  # 检查数据是否为空
        print(f"Warning: No data found in Visdom window '{win}'")
        return

    try:
        # 尝试解析窗口的数据
        window_json = json.loads(window_data)
        content = window_json.get('content', {})
        traces = content.get('data', [])  # 图表的曲线数据通常在 'data'

        # 创建绘图
        plt.figure(figsize=(10, 6))
        for trace in traces:
            x = trace['x']  # x 轴数据
            y = trace['y']  # y 轴数据
            name = trace.get('name', 'unknown')  # 获取曲线名称

            # 如果数据点的数量大于 1，才进行绘图
            if len(x) > 0 and len(y) > 0:
                plt.plot(x[1:], y[1:], label=name)  # 从第1个点开始绘图
                # plt.scatter(x[0:1], y[0:1], color='red')  # 单独画第0轮的点

                # 显示最后一个点的数值
                plt.annotate(
                    f'{y[-1]:.4f}',  # 只显示最后一个点的数值
                    (x[-1], y[-1]),  # 最后一个点的位置
                    textcoords="offset points",  # 偏移量模式
                    xytext=(0, 5),  # 偏移量（x方向和y方向）
                    ha='center',  # 水平对齐方式
                    fontsize=8,  # 字体大小
                    color='black'  # 字体颜色
                )
        # 获取标题
        title = content.get('layout', {}).get('title', {})
        if isinstance(title, dict) and 'text' in title:
            plt.title(title['text'])  # 如果是字典格式
        else:
            plt.title(title)  # 如果是字符串格式

        # 获取 x 和 y 轴的标签
        xaxis = content.get('layout', {}).get('xaxis', {}).get('title', {})
        plt.xlabel(xaxis['text'] if isinstance(xaxis, dict) and 'text' in xaxis else xaxis)

        yaxis = content.get('layout', {}).get('yaxis', {}).get('title', {})
        plt.ylabel(yaxis['text'] if isinstance(yaxis, dict) and 'text' in yaxis else yaxis)

        # 添加图例
        plt.legend()

        # 保存图表图片
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"Error while saving Visdom graph: {e}")


def create_folder_with_time(base_folder, current_time):
    """
    根据传入的时间字符串，在指定的 base_folder 下创建两个层级的文件夹：
    - 第一级：日期文件夹 (格式为 'YYYY_MM_DD')
    - 第二级：时间文件夹 (格式为 'HH_MM_SS')

    :param base_folder: 顶层文件夹名 (例如 'log' 或 'SCNet_surf1_img')
    :param current_time: 时间字符串，格式为 'YYYY_MM_DD_HH_MM_SS'
    :return: 返回最终创建的时间文件夹路径
    """
    try:
        # 检查时间字符串格式是否正确
        if "_" not in current_time:
            raise ValueError(f"Invalid time format: {current_time}. Expected format is 'YYYY_MM_DD_HH_MM_SS'.")

        # 拆分时间字符串为日期部分和时间部分
        parts = current_time.split("_")
        if len(parts) != 6:
            raise ValueError(f"Invalid time format: {current_time}. Expected format is 'YYYY_MM_DD_HH_MM_SS'.")

        date_part = "_".join(parts[:3])  # 日期部分 (YYYY_MM_DD)
        time_part = "_".join(parts[3:])  # 时间部分 (HH_MM_SS)

        # 构建日期文件夹路径
        date_folder_path = os.path.join(base_folder, date_part)

        # 创建日期文件夹（如果不存在）
        if not os.path.exists(date_folder_path):
            os.makedirs(date_folder_path)
            print(f"Created date folder: {date_folder_path}")

        # 构建时间文件夹路径
        time_folder_path = os.path.join(date_folder_path, time_part)

        # 创建时间文件夹（如果不存在）
        if not os.path.exists(time_folder_path):
            os.makedirs(time_folder_path)
            print(f"Created time folder: {time_folder_path}")

        # 返回最终创建的时间文件夹路径
        return time_folder_path

    except Exception as e:
        print(f"Error creating folder: {e}")
        return None


def preprocess_4d_tensor(input_tensor, device, repeat_times=None):
    """
    处理一个 4 维张量，标准化、调整通道顺序并转移到指定设备。

    参数：
        input_tensor: 输入的 4 维张量，形状为 (N, H, W, C)。
        device: 要将数据传输到的设备（例如 'cuda' 或 'cpu'）。
        repeat_times: 重复次数（可选），如果需要对张量进行维度重复。

    返回：
        在指定设备上的处理后的张量。
    """
    # 如果需要重复张量
    if repeat_times is not None:
        input_tensor = input_tensor.repeat(repeat_times, 1, 1, 1)

    # 标准化并转换数据类型
    processed_tensor = input_tensor.permute(0, 3, 1, 2).float().div(255)

    # 将数据转移到指定设备
    processed_tensor = processed_tensor.to(device)

    return processed_tensor


def preprocess_cam_surf_5_data(data_loader, device):
    # 获取一批数据
    data = next(iter(data_loader))

    # 转换到目标设备并调整数据类型
    data = data.to(device=device, dtype=torch.float32)

    # 调整维度并归一化
    data = data.permute(0, 3, 1, 2)
    data = data / 255.0

    # 使用 reshape 调整数据形状
    data = data.reshape(1, 15, 256, 256)

    return data


def load_state_dict_without_module(pth_file, model):
    """
    Load a state dict from a .pth file, remove 'module.' prefix from the keys,
    and load it into the given model.
    Parameters:
    pth_file (str): Path to the input .pth file.
    model (torch.nn.Module): The model to load the state dict into.
    """
    # Load the state dict
    state_dict = torch.load(pth_file)
    # Remove 'module.' prefix from the keys
    fixed_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # Load the fixed state dict into the model
    model.load_state_dict(fixed_state_dict, strict=False)
    print(f"{pth_file} State dict loaded into model successfully")


def flow_saveImgs(inputData, dir, start_idx=0):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if isinstance(inputData, torch.Tensor):
        if inputData.requires_grad:
            inputData = inputData.detach()
        if inputData.device.type == 'cuda':
            imgs = inputData.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
        else:
            imgs = inputData.numpy().transpose(0, 2, 3, 1)  # (N, C, row, col) to (N, row, col, C)
    else:
        imgs = inputData
    # imgs must have a shape of (N, row, col, C)
    imgs = np.uint8(imgs * 255)  # assume images are normalized to [0, 1]
    for i in range(imgs.shape[0]):
        file_name = 'img_{:04d}.png'.format(start_idx + i + 1)
        cv.imwrite(fullfile(dir, file_name), cv.cvtColor(imgs[i, :, :, :], cv.COLOR_BGR2RGB))


def warp_images(image, flow, mode='bilinear', padding_mode='zeros'):
    """
    返回经过mask过滤后的warp图，保证无效区域是0
    Args:
        image (torch.Tensor): (B, C, H, W)
        flow (torch.Tensor): (B, 2, H, W)
        mode (str): 'bilinear' or 'bicubic'
        padding_mode (str): 'zeros', 'border', 'reflection'
    Returns:
        warped_clean (torch.Tensor): (B, C, H, W)
    """
    B, C, H, W = image.size()
    device = image.device
    # 生成标准网格
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1.0, 1.0, W, device=device),
        torch.linspace(-1.0, 1.0, H, device=device),
        indexing='xy'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
    # 归一化 flow
    flow_norm = torch.zeros_like(flow)
    flow_norm[:, 0, :, :] = flow[:, 0, :, :] / ((W - 1) / 2)
    flow_norm[:, 1, :, :] = flow[:, 1, :, :] / ((H - 1) / 2)
    flow_norm = flow_norm.permute(0, 2, 3, 1)
    vgrid = grid + flow_norm
    # grid_sample
    warped = F.grid_sample(image, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
    # 同样grid_sample一个全1图像，拿到mask
    # ones = torch.ones_like(image[:, :1, :, :])  # (B, 1, H, W)
    # mask = F.grid_sample(ones, vgrid, mode='nearest', padding_mode='zeros')
    # mask = (mask >= 0.999).float()
    # 用mask清除无效区域
    # warped_clean = warped * mask
    # if return_all:
    return warped


#
# def warp_images(image, flow, mode='bilinear', padding_mode='zeros', return_all=False):
#     """
#     使用光流 warp 图像，返回：
#         - warped: 原始 flow 的 warp 图像
#         - mask: 有效 flow 区域（不越界）
#         - warped_clean: 掩掉越界 flow 后 warp 的图像
#     """
#     B, C, H, W = image.size()
#     device = image.device
#
#     # 生成标准归一化网格 base_grid，[-1, 1]
#     grid_x, grid_y = torch.meshgrid(
#         torch.linspace(-1.0, 1.0, W, device=device),
#         torch.linspace(-1.0, 1.0, H, device=device),
#         indexing='xy'
#     )
#     base_grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
#     base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
#
#     # 归一化 flow → [-1, 1] 坐标系下的 flow
#     flow_norm = torch.zeros_like(flow)
#     flow_norm[:, 0] = flow[:, 0] / ((W - 1) / 2)
#     flow_norm[:, 1] = flow[:, 1] / ((H - 1) / 2)
#     flow_norm = flow_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)
#
#     # 计算目标 grid（未mask）
#     vgrid = base_grid + flow_norm
#
#     # ----- 1. 原始 flow 扭正图像 -----
#     warped = F.grid_sample(image, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
#
#     # ----- 2. 根据 vgrid 越界情况生成 mask -----
#     mask_x = (vgrid[..., 0] >= -1.0) & (vgrid[..., 0] <= 1.0)
#     mask_y = (vgrid[..., 1] >= -1.0) & (vgrid[..., 1] <= 1.0)
#     mask = (mask_x & mask_y).float()  # (B, H, W)
#
#     # 将 mask 转成 (B, 1, H, W)，以便与 image 通道数一致
#     mask_unsq = mask.unsqueeze(1)  # (B, 1, H, W)
#
#     # ----- 3. 用 mask 过滤 flow（无效位置 flow = 0） -----
#     masked_flow = flow * mask_unsq
#
#     # 归一化 masked flow
#     flow_masked_norm = torch.zeros_like(masked_flow)
#     flow_masked_norm[:, 0] = masked_flow[:, 0] / ((W - 1) / 2)
#     flow_masked_norm[:, 1] = masked_flow[:, 1] / ((H - 1) / 2)
#     flow_masked_norm = flow_masked_norm.permute(0, 2, 3, 1)
#
#     # 计算 masked flow 的目标 grid
#     vgrid_clean = base_grid + flow_masked_norm
#
#     # ----- 4. 用 masked flow warp 图像（真正 clean 的扭正）-----
#     warped_clean = F.grid_sample(image, vgrid_clean, mode=mode, padding_mode=padding_mode, align_corners=True)
#
#     if return_all:
#         return warped, mask_unsq, warped_clean
#     else:
#         return warped_clean

def save_config_and_model_info(config, model, output_dir):
    """
    保存配置文件和模型的相关信息到指定文件夹中的日志文件。

    Args:
        config (CN): 配置对象（如 _CN）。
        model (object): 已声明的外部模型对象。
        output_dir (str): 保存日志文件的文件夹路径。
    """
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 定义日志文件名
    output_file_path = os.path.join(output_dir, "config_and_model_info.txt")

    # 获取模型的类定义源码
    cls = model.__class__
    class_source_code = inspect.getsource(cls)

    # 写入配置和模型信息到同一个文件
    with open(output_file_path, 'w') as output_file:
        # 写入配置内容
        output_file.write("### Configuration ###\n")
        output_file.write(config.dump())  # 将配置的内容以 YAML 格式写入文件
        output_file.write("\n\n")

        # 写入模型信息
        output_file.write("### Model Information ###\n")
        output_file.write(f"Model Name: {model.__class__.__name__}\n\n")
        output_file.write("### Model Class Definition ###\n")
        output_file.write(class_source_code)

    print(f"Configuration and model information saved to: {output_file_path}")


def get_scheduler(optimizer, scheduler_config):
    """
    根据配置返回相应的学习率调度器
    """
    scheduler_type = scheduler_config.type
    if scheduler_type == 'step_lr':
        return StepLR(optimizer, step_size=scheduler_config.step_size, gamma=scheduler_config.gamma)
    elif scheduler_type == 'lambda_lr':
        a = scheduler_config.a
        lambda_func = lambda it: 1 / (1 + a * it)
        return LambdaLR(optimizer, lr_lambda=lambda_func)
    elif scheduler_type == 'cosine_annealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config.T_max, eta_min=scheduler_config.eta_min)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def save_two_checkpoint(model_a, model_b, optimizer_a, optimizer_b, scheduler_a, scheduler_b, total_steps, save_interval, save_dir,
                        is_final=False):
    """
    保存模型的检查点

    参数：
    - model_a: 要保存的模型 A
    - model_b: 要保存的模型 B
    - optimizer_a: 模型 A 的优化器
    - optimizer_b: 模型 B 的优化器
    - scheduler_a: 模型 A 的学习率调度器
    - scheduler_b: 模型 B 的学习率调度器
    - total_steps: 当前的迭代次数
    - save_interval: 每多少轮保存一次
    - save_dir: 保存的基础目录
    - current_time: 当前时间字符串，用于创建独立的文件夹
    - is_final: 是否为最终保存（布尔值）
    """

    # 判断是否需要保存检查点
    should_save = (total_steps % save_interval == 0) or is_final

    if should_save:
        # 根据是否为最终保存，决定文件名
        if is_final:
            filename = f'step_{total_steps}_final.pth'
        else:
            filename = f'step_{total_steps}.pth'

        # 定义保存的文件路径
        final_path = os.path.join(save_dir, filename)

        try:
            # 创建一个字典，将两个模型、优化器和调度器的状态保存
            checkpoint = {
                'model_a_state_dict': model_a.module.state_dict() if isinstance(model_a, DataParallel) else model_a.state_dict(),
                'model_b_state_dict': model_b.module.state_dict() if isinstance(model_b, DataParallel) else model_b.state_dict(),
                'optimizer_a_state_dict': optimizer_a.state_dict(),
                'optimizer_b_state_dict': optimizer_b.state_dict(),
                'scheduler_a_state_dict': scheduler_a.state_dict(),
                'scheduler_b_state_dict': scheduler_b.state_dict(),
                'total_steps': total_steps,
            }

            # 保存检查点
            torch.save(checkpoint, final_path)
            print(f"保存检查点到: {final_path}")
        except Exception as e:
            print(f"保存检查点时发生错误: {e}")


def load_checkpoint(filename, model_a, model_b, optimizer_a, optimizer_b, scheduler_a, scheduler_b):
    """
    加载检查点

    参数：
    - filename: 保存的检查点文件路径
    - model_a: 要加载的模型 A
    - model_b: 要加载的模型 B
    - optimizer_a: 模型 A 的优化器
    - optimizer_b: 模型 B 的优化器
    - scheduler_a: 模型 A 的学习率调度器
    - scheduler_b: 模型 B 的学习率调度器

    返回：
    - total_steps: 加载的总迭代次数
    """
    try:
        # 加载检查点
        checkpoint = torch.load(filename)

        # 加载模型状态字典
        model_a.load_state_dict(checkpoint['model_a_state_dict'])
        model_b.load_state_dict(checkpoint['model_b_state_dict'])

        # 加载优化器状态字典
        optimizer_a.load_state_dict(checkpoint['optimizer_a_state_dict'])
        optimizer_b.load_state_dict(checkpoint['optimizer_b_state_dict'])

        # 加载学习率调度器状态字典
        scheduler_a.load_state_dict(checkpoint['scheduler_a_state_dict'])
        scheduler_b.load_state_dict(checkpoint['scheduler_b_state_dict'])

        # 返回加载的总迭代次数
        total_steps = checkpoint['total_steps']
        print(f"成功加载检查点: {filename} (总迭代次数: {total_steps})")
        return total_steps

    except Exception as e:
        print(f"加载检查点时发生错误: {e}")
        raise


def save_one_model_checkpoint(model, total_steps, save_interval, save_dir, current_time, is_final=False):
    """
    保存模型的检查点

    参数：
    - model: 要保存的模型
    - total_steps: 当前的迭代次数
    - save_interval: 每多少轮保存一次
    - save_dir: 保存的基础目录
    - current_time: 当前时间字符串，用于创建独立的文件夹
    - is_final: 是否为最终保存（布尔值）
    """

    # 判断是否需要保存检查点
    should_save = (total_steps % save_interval == 0) or is_final

    if should_save:
        # 创建新的保存目录（基于当前时间）
        checkpoint_dir = os.path.join(save_dir, current_time)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 根据是否为最终保存，决定文件名
        if is_final:
            filename = f'step_{total_steps}_final.pth'
        else:
            filename = f'step_{total_steps}.pth'

        # 定义保存的文件路径
        final_path = os.path.join(checkpoint_dir, filename)

        try:
            # 如果模型使用了 DataParallel，保存其 module 的 state_dict
            if isinstance(model, DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存模型的状态字典
            torch.save(state_dict, final_path)
            print(f"保存检查点到: {final_path}")
        except Exception as e:
            print(f"保存检查点时发生错误: {e}")


def import_datasets_module(choice):
    """
    根据用户的选择导入相应的 datasets_lists 模块。

    Args:
        choice (str): 用户的选择，应该是 'lab2' 或 'lab3'。

    Returns:
        module: 导入的模块。
    """
    if choice.lower() == 'lab2':
        try:
            import configs.lab2_datasets_lists as datasets_module
            print("已成功导入模块 configs.lab2_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.lab2_datasets_lists 失败: {e}")
            sys.exit(1)

    elif choice.lower() == 'lab3':
        try:
            import configs.lab3_datasets_lists as datasets_module
            print("已成功导入模块 configs.lab3_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.lab3_datasets_lists 失败: {e}")
            sys.exit(1)
    elif choice.lower() == 'local':
        try:
            import configs.local_datasets_lists_1218 as datasets_module
            print("已成功导入模块 configs.local_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.local_datasets_lists 失败: {e}")
            sys.exit(1)
    elif choice.lower() == 'sl_lab3':
        try:
            import configs.SL_lab3_datasets_lists as datasets_module
            print("已成功导入模块 configs.SL_lab3_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.SL_lab3_datasets_lists 失败: {e}")
            sys.exit(1)

    elif choice.lower() == 'sl_lab2':
        try:
            import configs.SL_lab2_datasets_lists as datasets_module
            print("已成功导入模块 configs.SL_lab2_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.SL_lab2_datasets_lists 失败: {e}")
            sys.exit(1)

    elif choice.lower() == 'sl_local':
        try:
            import configs.SL_local_datasets_lists as datasets_module
            print("已成功导入模块 configs.SL_local_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.SL_local_datasets_lists 失败: {e}")
            sys.exit(1)

    elif choice.lower() == 'sl_without_cmp_lab3':
        try:
            import configs.SL_lab3_datasets_lists_without_cmp as datasets_module
            print("已成功导入模块 configs.SL_local_datasets_lists")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.SL_local_datasets_lists 失败: {e}")
            sys.exit(1)

    elif choice.lower() == 'sl_without_cmp_local':
        try:
            import configs.SL_local_datasets_lists_without_cmp as datasets_module
            print("已成功导入模块 configs.SL_local_datasets_lists_without_cmp")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.SL_local_datasets_lists_without_cmp 失败: {e}")
            sys.exit(1)

    elif choice.lower() == 'test_local':
        try:
            import configs.Test_datasets as datasets_module
            print("已成功导入模块 configs.Test_local")
            return datasets_module
        except ImportError as e:
            print(f"导入模块 configs.Test_local 失败: {e}")
            sys.exit(1)
    else:
        print(f"未知的 dataset_type: '{choice}'. 请使用 'lab2' 或 'lab3'.")
        sys.exit(1)


def visdom_display_save(vis, cam_crop, warped_predict, prj_GT, batch_size, step, phase='train',
                        max_nrow=5, save_path=None, win=None):
    """
    可视化拼接后的图像并显示在 Visdom 窗口中，同时可选择将图像保存到本地文件。

    Args:
        vis (visdom.Visdom): 已初始化的 Visdom 客户端。
        cam_crop (torch.Tensor): 裁剪后的相机图像张量，形状为 (batch_size, C, H, W)。
        warped_predict (torch.Tensor): 扭曲预测图像张量，形状为 (batch_size, C, H, W)。
        prj_GT (torch.Tensor): Ground Truth 投影图像张量，形状为 (batch_size, C, H, W)。
        batch_size (int): 批量大小。
        step (int): 当前的训练或验证步骤。
        phase (str, optional): 当前阶段，'train' 或 'valid'。默认为 'train'。
        max_nrow (int, optional): 每行的最大图像数量。默认为5。
        win (str, optional): Visdom 窗口的唯一标识符。如果为None，将自动生成。
        save_path (str, optional): 本地保存图像的路径。如果为None，则不保存。默认为None。

    Returns:
        None
    """
    with torch.no_grad():
        # 判断 batch_size 并选择前 max_nrow 个样本
        if batch_size > max_nrow:
            # print(f"Batch size {batch_size} > {max_nrow}, 选择前 {max_nrow} 个样本进行可视化。")
            cam_crop = cam_crop[:max_nrow]
            warped_predict = warped_predict[:max_nrow]
            prj_GT = prj_GT[:max_nrow]
        else:
            print(f"Batch size {batch_size} <= {max_nrow}, 使用全部 {batch_size} 个样本进行可视化。")

        # 拼接图像张量
        images = torch.cat([cam_crop, warped_predict, prj_GT], dim=0)
        # print(f"拼接后的图像形状: {images.shape}")  # 预期形状 (3 * selected_count, C, H, W)

        # 创建图像网格
        nrow = min(batch_size, max_nrow)
        images_grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
        # print(f"图像网格形状: {images_grid.shape}")  # 预期形状 (C, H, W)

        # 生成窗口ID，如果未提供
        if win is None:
            win = f"{phase}_{step}"

        # 设置标题
        full_title = f"{phase.capitalize()}_Step: {step}"

        # 显示图像在 Visdom 中
        vis.image(
            images_grid.cpu().numpy(),  # 无需交换维度
            opts=dict(title=full_title),
            win=win
        )
        # print(f"已在 Visdom 窗口 '{win}' 中显示: {full_title}")

        # 保存图像到本地文件（如果提供了 save_path）
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存目录存在
            vutils.save_image(images, save_path, nrow=nrow, normalize=True, scale_each=True)
            # print(f"图像已保存到: {save_path}"


def log_test_metrics(logger, step, metrics):
    """
    记录测试指标到 WandB。
    Parameters:
    - logger (WandBLogger): WandB 记录器实例
    - step (int): 当前步骤，用于作为横坐标
    - metrics (dict): 需要记录的指标字典，包含 'test_mae', 'test_ssim', 'test_psnr', 'test_rmse', 'test_deltaE', 'test_lpips'
    """
    # 记录每个单独的指标
    for key, value in metrics.items():
        logger.log_metrics(step, {key: value})

        # 计算总损失并记录
    total_loss = (
            metrics['test_mae'] +
            metrics['test_ssim'] +
            metrics['test_rmse'] +
            metrics['test_deltaE'] +
            metrics['test_lpips']
    )
    logger.log_metrics(step, {'test_loss': total_loss})
