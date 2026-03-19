import copy
import cv2 as cv,cv2
import numpy as np
from torch import nn
import ctypes
import pytorch_ssim

l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM()
import os
import glob
import sys

def save_mask(mask, folder_path, filename=None, overwrite=False, dsize=None):
    """
    保存二值化 Mask 图像（bool 类型或 0/1），转换为 0/255 单通道图像后保存。

    @param mask: numpy 数组，类型为 bool, 0/1 或 0/255 的单通道或三通道 mask。
    @param folder_path: 保存路径。
    @param filename: 自定义文件名，不带扩展名时自动补 .png。
    @param overwrite: 是否覆盖已有文件。
    @param dsize: 保存的尺寸 (width, height)，默认不缩放。
    @return: 保存的完整路径。
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 处理 mask 数据为 0 或 255 的 uint8 单通道图像
    if mask.dtype == np.bool_:
        mask = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)

    if mask.ndim == 3 and mask.shape[2] > 1:
        mask = mask[:, :, 0]  # 保证单通道

    # 缩放（如指定）
    if dsize is not None:
        mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)

    # 生成文件名
    if filename:
        name, ext = os.path.splitext(filename)
        if not ext:
            ext = '.png'
        new_filename = name + ext
    else:
        new_filename = "mask_0001.png"
        if not overwrite:
            i = 1
            while os.path.exists(os.path.join(folder_path, new_filename)):
                new_filename = f"mask_{i:04d}.png"
                i += 1

    full_path = os.path.join(folder_path, new_filename)

    if not overwrite and os.path.exists(full_path):
        base, ext = os.path.splitext(new_filename)
        j = 1
        while os.path.exists(os.path.join(folder_path, f"{base}_{j:03d}{ext}")):
            j += 1
        new_filename = f"{base}_{j:03d}{ext}"
        full_path = os.path.join(folder_path, new_filename)

    # 保存
    cv2.imwrite(full_path, mask)
    return full_path

# def thresh(im_in):
#     # threshold im_diff with Otsu's method
#     if im_in.ndim == 3:
#         im_in = cv.cvtColor(im_in, cv.COLOR_BGR2GRAY)
#     if im_in.dtype == 'float32':
#         im_in = np.uint8(im_in * 255)
#     _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#     im_mask = im_mask > 0
#
#     # find the largest contour by area then convert it to convex hull
#     contours, hierarchy = cv.findContours(np.uint8(im_mask), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     hulls = cv.convexHull(max(contours, key=cv.contourArea))
#     im_mask = cv.fillConvexPoly(np.zeros_like(im_mask, dtype=np.uint8), hulls, True) > 0
#
#     # also calculate the bounding box
#     bbox = cv.boundingRect(max(contours, key=cv.contourArea))
#     corners = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]], [bbox[0], bbox[1] + bbox[3]]]
#
#     # normalize to (-1, 1) following pytorch grid_sample coordinate system
#     h = im_in.shape[0]
#     w = im_in.shape[1]
#
#     for pt in corners:
#         pt[0] = 2 * (pt[0] / w) - 1
#         pt[1] = 2 * (pt[1] / h) - 1
#
#     return im_mask, corners


def thresh(im_in):
    # Convert to grayscale if needed
    if im_in.ndim == 3:
        im_in = cv.cvtColor(im_in, cv.COLOR_BGR2GRAY)
    if im_in.dtype == 'float32':
        im_in = np.uint8(im_in * 255)
    # Otsu thresholding with Gaussian blur
    _, im_mask = cv.threshold(cv.GaussianBlur(im_in, (5, 5), 0), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Morphological closing to fill small holes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    im_mask = cv.morphologyEx(im_mask, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(im_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(im_mask, dtype=bool), []

    # Use the largest contour directly (no convex hull)
    largest_contour = max(contours, key=cv.contourArea)

    # Fill the contour itself (not convex hull)
    im_mask_filled = np.zeros_like(im_mask, dtype=np.uint8)
    cv.drawContours(im_mask_filled, [largest_contour], contourIdx=-1, color=1, thickness=-1)
    im_mask_bool = im_mask_filled > 0

    # Bounding box of the contour
    bbox = cv.boundingRect(largest_contour)
    corners = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]]
    ]
    # Normalize corners to (-1, 1)
    h, w = im_in.shape[:2]
    for pt in corners:
        pt[0] = 2 * (pt[0] / w) - 1
        pt[1] = 2 * (pt[1] / h) - 1
    return im_mask_bool, corners


def visualize_mask_with_opencv(mask: np.ndarray, window_name: str = "Mask"):
    """
    使用 OpenCV 显示一个 mask 图像。

    参数:
        mask (np.ndarray): 二值化的掩膜图像，可以是布尔、uint8 或 float 格式。
        window_name (str): 窗口显示名称。
    """
    # 确保 mask 是 uint8 类型，范围在 [0, 255]
    if mask.dtype == np.bool_:
        mask_vis = (mask.astype(np.uint8)) * 255
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        mask_vis = np.clip(mask * 255, 0, 255).astype(np.uint8)
    elif mask.dtype == np.uint8:
        mask_vis = mask.copy()
    else:
        raise TypeError(f"Unsupported mask dtype: {mask.dtype}")

    # 如果是单通道就直接显示
    if mask_vis.ndim == 2:
        pass
    elif mask_vis.ndim == 3 and mask_vis.shape[2] == 1:
        mask_vis = mask_vis[:, :, 0]
    else:
        raise ValueError(f"Expected single-channel mask, got shape {mask.shape}")

    # 可视化
    cv.imshow(window_name, mask_vis)
    cv.waitKey(0)
    cv.destroyAllWindows()


def compute_mask_from_paths(path1, path2):
    """
    读取 path1 和 path2 下的图像，计算图像差，并用 thresh 提取掩膜与角点。

    Args:
        path1 (str): 原始图像路径（例如背景图）
        path2 (str): 目标图像路径（例如有前景的图）

    Returns:
        im_mask (np.ndarray): 掩膜（bool 类型，True 表示前景区域）
        corners (list of list): 归一化后的四个角点坐标，顺序为 [左上, 右上, 右下, 左下]
    """

    # 读取图像
    img1 = cv.imread(path1, cv.IMREAD_COLOR)
    img2 = cv.imread(path2, cv.IMREAD_COLOR)

    # 安全性检查
    if img1 is None:
        raise FileNotFoundError(f"Cannot read image at {path1}")
    if img2 is None:
        raise FileNotFoundError(f"Cannot read image at {path2}")

    # 将图像转为灰度
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 计算差分图像（取绝对值）
    diff = cv.absdiff(gray2, gray1)

    # 调用 thresh 函数
    im_mask, corners = thresh(diff)

    return im_mask, corners

def delete_images_in_folder(folder_path):
    """
    删除指定文件夹中的所有图像文件（不删除文件夹本身）。

    参数:
        folder_path (str): 要清空图片的文件夹路径
    """
    # 支持的图像扩展名
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    files = os.listdir(folder_path)
    deleted_count = 0

    for filename in files:
        if filename.lower().endswith(valid_exts):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败: {filename}，错误信息: {e}")

    print(f"🧹 已删除 {deleted_count} 张图像文件，来自：{folder_path}")
def compute_direct_indirect_light(im_cb, backlight_strength=0.9):
    """
    计算直接光与间接光分量（基于Nayar TOG'06方法，参考Moreno 3DV'12）。

    参数:
        im_cb: np.ndarray
            相机拍摄的checkerboard图像序列，shape应为(H, W, C, N)
            或 torch.Tensor，shape为(N, C, H, W)
        backlight_strength: float
            投影仪背光强度，默认值为0.9（用于mask时建议高一些）

    返回:
        im_direct: np.ndarray
            直接光图像 (H, W, C)
        im_indirect: np.ndarray
            间接光图像 (H, W, C)
    """
    # 如果是 torch tensor，先转为 numpy 格式并调整维度
    if hasattr(im_cb, 'numpy'):
        im_cb = im_cb.numpy().transpose((2, 3, 1, 0))  # (H, W, C, N)

    l1 = np.max(im_cb, axis=3)  # 最大图像 L+
    l2 = np.min(im_cb, axis=3)  # 最小图像 L-

    b = backlight_strength
    im_direct = (l1 - l2) / (1 - b)  # 直接光
    im_indirect = 2 * (l2 - b * l1) / (1 - b ** 2)  # 间接光
    im_indirect = np.clip(im_indirect, a_min=0.0, a_max=None)

    # 若间接光为负值，直接光直接使用L+
    im_direct[im_indirect < 0] = l1[im_indirect < 0]

    return im_direct, im_indirect

def check_image_folder_exists(folder_path, verbose=True, exit_on_fail=True):
    """
    检查指定文件夹中是否存在图像文件。

    参数:
        folder_path (str): 要检查的文件夹路径。
        verbose (bool): 是否打印日志信息。
        exit_on_fail (bool): 如果没有找到图像文件，是否退出程序。

    返回:
        List[str]: 所有找到的图像文件路径列表。
    """
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        msg = f"❌ Error: No image files found in {folder_path}."
        if verbose:
            print(msg)
        if exit_on_fail:
            sys.exit(1)
        else:
            raise FileNotFoundError(msg)
    else:
        if verbose:
            print(f"✅ Found {len(image_files)} image(s) in {folder_path}.")

    return image_files


def checkerboard(*args):
    # Python implementation of MATLAB's checkerboard function
    # Parse inputs
    n = 10
    p = 4
    q = p

    if len(args) == 1:
        n = args[0]
    elif len(args) == 2:
        n, p = args
        q = p
    elif len(args) == 3:
        n, p, q = args

    # Generate tile
    tile = np.tile(np.kron([[0, 1], [1, 0]], np.ones((n, n))), (1, 1))

    # Create checkerboard
    if q % 2 == 0:
        # Make left and right sections separately
        num_col_reps = int(np.ceil(q / 2))
        ileft = np.tile(tile, (p, num_col_reps))

        tile_right = np.tile(np.kron([[0, 0.7], [0.7, 0]], np.ones((n, n))), (1, 1))
        iright = np.tile(tile_right, (p, num_col_reps))

        # Tile the left and right halves together
        checkerboard = np.concatenate((ileft, iright), axis=1)
    else:
        # Make the entire image in one shot
        checkerboard = np.tile(tile, (p, q))

        # Make right half plane have light gray tiles
        mid_col = int(checkerboard.shape[1] / 2) + 1
        checkerboard[:, mid_col:] = checkerboard[:, mid_col:] - .3
        checkerboard[np.where(checkerboard < 0)] = 0

    return checkerboard.astype('float64')

def load_images_from_folder(folder_path):
    """
    从指定文件夹中读取所有图像，并将其打包成四维变量，格式为 N x H x W x C
    @param folder_path: 包含图片的文件夹路径
    @return: 一个四维的 numpy 数组，形状为 N x H x W x C
    """
    image_list = []
    common_height = None
    common_width = None
    common_channels = None

    # 获取文件夹下所有文件名
    files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # # 将BGR转换为RGB
            # SCNet_surf1_img = cv2.cvtColor(SCNet_surf1_img, cv2.COLOR_BGR2RGB)

            if common_height is None:
                # 第一张图片，确定基本的尺寸和通道数
                common_height, common_width, common_channels = img.shape
            else:
                # 其他图片，需要检查尺寸和通道数
                h, w, c = img.shape
                if h != common_height or w != common_width or c != common_channels:
                    print(f"Resizing image {filename} to common size ({common_height}, {common_width}, {common_channels}).")
                    img = cv2.resize(img, (common_width, common_height))

            image_list.append(img)
        else:
            print(f"Warning: Unable to read image {img_path}")

    if len(image_list) == 0:
        print("no image in the folder")

    # 将所有图片堆叠成四维数组
    image_array = np.stack(image_list, axis=0)

    return image_array


def save_correspondences_to_yml(cam_pts, prj_pts, filename='sl.yml'):
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    fs.write("cam_pts", cam_pts)
    fs.write("prj_pts", prj_pts)
    print("cam_pts min:", np.min(cam_pts, axis=0))
    print("cam_pts max:", np.max(cam_pts, axis=0))
    print("prj_pts min:", np.min(prj_pts, axis=0))
    print("prj_pts max:", np.max(prj_pts, axis=0))
    fs.release()


class Camera:
    def __init__(self, camera_index=0, window_name='Camera Preview, Press q to exit', delay_frames=5, delay_time=0.2):
        """
        初始化 Camera 对象。

        参数:
        - camera_index: 整数，指定要使用的摄像头的索引（默认值为0）。
        - window_name: 字符串，指定显示窗口的名称（默认值为 'Camera Preview, Press q to exit'）。
        - delay_frames: 整数，指定捕捉最新图像之前要丢弃的帧数（默认值为5）。
        - delay_time: 浮点数，指定投影与捕捉之间的时间间隔，以秒为单位（默认值为0.1秒）。
        """
        self.camera_index = camera_index  # 摄像头索引
        self.window_name = window_name  # 窗口名称
        self.delay_frames = delay_frames  # 要丢弃的帧数
        self.delay_time = delay_time  # 投影与捕捉之间的时间间隔
        self.cap = cv2.VideoCapture(camera_index)  # 打开指定索引的摄像头
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")  # 如果摄像头无法打开，抛出异常

    def clear_buffer(self):
        """
        清空相机缓冲区，通过读取并丢弃多帧图像来实现。
        """
        for _ in range(20):
            self.cap.read()

    def capturing(self):
        """
        从摄像头捕获最新帧，丢弃前几个延迟帧。

        返回值:
        - frame: 捕获的图像帧。

        抛出:
        - RuntimeError: 如果无法读取摄像头数据。
        """
        # 等待一定的时间
        # time.sleep(self.delay_time)

        # 读取并丢弃指定数量的延迟帧
        # for _ in range(self.delay_frames):
        #     self.cap.read()

        # 捕获图像
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法读取摄像头数据")  # 如果读取失败，则抛出异常
        return frame  # 返回捕获的图像帧

    def preview(self):
        """
        实时显示摄像头图像，直到按下 'q' 键退出。
        """
        while True:
            ret, frame = self.cap.read()  # 捕获一帧图像
            if not ret:
                raise RuntimeError("无法读取摄像头数据")

            # 在窗口中显示捕获的图像
            cv2.imshow(self.window_name, frame)

            # 检查是否按下 'q' 键，如果按下则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()

    def release(self):
        """
        释放摄像头和关闭所有 OpenCV 窗口。
        """
        # 读取并丢弃指定数量的延迟帧
        for _ in range(self.delay_frames):
            self.cap.read()
        self.cap.release()  # 释放摄像头
        cv2.destroyAllWindows()


class Projecting:
    def __init__(self, sw, sh):
        """
        初始化Projecting类的实例。

        @param sw: 屏幕宽度
        @param sh: 屏幕高度
        """
        super().__init__()
        self.prj_win = 'full_screen'
        self.sw, self.sh = sw, sh
        cv2.namedWindow(self.prj_win, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.prj_win, 2561, 0)
        cv2.setWindowProperty(self.prj_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.img = np.zeros((self.sh, self.sw, 3), dtype=np.uint8)

    def projecting(self, img):
        """
        在全屏窗口中投影图像。

        @param img: 要显示的图像
        """
        img_copy = img.copy()

        if img_copy.max() <= 1.0 and img_copy.min() >= 0.0:
            img_copy = (img_copy * 255).astype(np.uint8)

        img_h, img_w = img_copy.shape[:2]
        screen_ratio = self.sw / self.sh
        img_ratio = img_w / img_h

        if abs(img_ratio - 1.0) < 0.01:  # 图片是1:1比例
            # 按高度调整图像大小
            new_h = self.sh
            new_w = self.sh
        else:  # 图片不是1:1比例
            new_h = self.sh
            new_w = int(self.sh * img_ratio)

        # 确保图像宽度或高度不超过屏幕尺寸
        if new_w > self.sw:
            new_w = self.sw
            new_h = int(self.sw / img_ratio)

        if new_h > self.sh:
            new_h = self.sh
            new_w = int(self.sh * img_ratio)

        img_copy = cv2.resize(img_copy, (new_w, new_h))
        h, w = img_copy.shape[:2]

        start_x = (self.sw - w) // 2
        start_y = (self.sh - h) // 2

        # self.SCNet_surf1_img.fill(0)  # 清空之前的显示内容
        self.img[start_y:start_y + h, start_x:start_x + w, :3] = img_copy[:, :, :]

        cv2.imshow(self.prj_win, self.img)
        cv2.waitKey(200)


def normalize_image(img):
    """
    检查图像是否已经归一化，如果没有归一化则进行归一化处理。

    @param img: 输入图像 (numpy array)
    @return: 归一化后的图像
    """
    # 检查图像的最大值是否大于1，如果是，则认为图像未归一化，进行归一化处理
    if img.max() > 1.0:
        img = img.astype(float) / 255.0

    return img

def extract_filename(file_path):
    """
    从文件路径中提取文件名。

    参数:
    file_path : 文件完整路径 (字符串)

    返回:
    文件名 (字符串)
    """
    return os.path.basename(file_path)
def read_and_normalize_image(image_path, normalize=True):
    """
    读取图像文件，并选择性地进行归一化处理。

    @param image_path: 图像文件路径
    @param normalize: 如果为 True，则将图像归一化到0-1范围，默认为 True

    @return: 读取并处理过的图像
    """
    # 读取图像文件
    img = cv2.imread(image_path)

    # 检查图像是否成功读取
    if img is None:
        raise FileNotFoundError(f"Unable to load image from file: {image_path}")

    # 如果需要归一化，则将图像归一化到0-1范围
    if normalize:
        img = img.astype(float) / 255.0

    return img


def clamp_image(image, min_value=0.0, max_value=1.0):
    """

    @param image: 格式是opencv读取图片的格式
    @param min_value: 0
    @param max_value: 1
    @return: 返回clamp的输出，因为最后计算的图像会超过1，所以要clamp
    """
    return np.clip(image, min_value, max_value)

def create_directory_structure(base_path,current_time):
    """
    在指定的根目录下创建一组文件夹结构，并返回创建的路径字典。

    @param base_path: 根目录路径
    @return: 包含创建的文件夹路径的字典
    """
    # 定义需要创建的文件夹结构
    directories = {
        "mask": f"{current_time}/cam/mask",
        # "cam_raw_cb": f"{current_time}/cam/raw/cb",
        "cam_raw_train": f"{current_time}/cam/raw/train",
        "cam_raw_test": f"{current_time}/cam/raw/test",
        "cam_raw_ref": f"{current_time}/cam/raw/ref",
        'cam_raw_sl': f"{current_time}/cam/raw/sl",
        "cam_warp_train": f"{current_time}/cam/warpSL/train",
        "cam_warp_test": f"{current_time}/cam/warpSL/test",
        "cam_warp_ref": f"{current_time}/cam/warpSL/ref",
    }
    created_directories = {}

    for key, directory in directories.items():
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            created_directories[key] = dir_path
            print(f"Created directory: {dir_path}")
        else:
            created_directories[key] = dir_path
            print(f"Directory already exists: {dir_path}")

    return created_directories

def save_images(image_data, folder_path, filename=None, overwrite=False, dsize=None):
    """
    这个函数是保存图像的函数，这个函数的输入是opencv读取的数据，范围要求 0-255。
    如果overwrite = False, 假设已经有了相同文件名的图片，那么就会保存为新的img_000x_yyy.png
    如果overwrite = True，那么文件只会覆盖保存为指定的文件名或默认的img_0001.png。
    可以选择保存的图像大小。

    @param image_data: opencv读取的图像数据，可以是三维或四维数组
    @param folder_path: 保存的文件路径
    @param filename: 自定义保存的文件名（不包含扩展名）。默认为 None。
    @param overwrite: 是否覆盖现有文件，默认为 False
    @param dsize: 要调整为的图像大小，格式为 (width, height)。默认为 None。
    @return: 返回保存的文件名列表
    """
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 确定图像数据是否为四维，如果是，要遍历处理
    is_four_dim = len(image_data.shape) == 4

    if not is_four_dim:
        image_data = image_data[None, ...]  # 将三维数据包装成四维，方便处理

    # 检查图像数据的范围，如果在0-1之间，则乘以255
    if image_data.min() >= 0 and image_data.max() <= 1:
        image_data = (image_data * 255).astype('uint8')
    else:
        image_data = image_data.astype('uint8')

    saved_filenames = []

    for i in range(image_data.shape[0]):  # 遍历每帧图像
        frame = image_data[i]

        # 如果需要调整图像大小
        if dsize is not None:
            frame = cv2.resize(frame, dsize)

        # 如果提供了自定义文件名
        if filename:
            name, ext = os.path.splitext(filename)
            if not ext:  # 如果用户没提供扩展名，默认使用png
                ext = '.png'
            new_filename = name + ext
        else:
            # 如果覆盖选项为 True，或第一次调用，使用固定的起始名称
            if overwrite or i == 0:
                new_filename = f'img_{i + 1:04d}.png'
            else:
                # 获取文件夹下已存在的图片文件名列表
                existing_files = [f for f in os.listdir(folder_path) if
                                  os.path.isfile(os.path.join(folder_path, f)) and f.startswith('img_') and f.endswith('.png')]

                if existing_files:
                    # 获取最大的文件编号
                    max_existing_num = max(int(f.split('_')[1].split('.')[0]) for f in existing_files if '_' in f)
                    file_number = max_existing_num + 1
                    new_filename = f'img_{file_number:04d}_{i + 1:03d}.png'
                else:
                    new_filename = f'img_{i:04d}_{i + 1:03d}.png'

        # 拼接文件地址
        full_path = os.path.join(folder_path, new_filename)

        # 检查是否要覆盖已有文件
        if not overwrite and os.path.exists(full_path):
            base, ext = os.path.splitext(new_filename)
            j = 1
            while os.path.exists(os.path.join(folder_path, f"{base}_{j:03d}{ext}")):
                j += 1
            new_filename = f"{base}_{j:03d}{ext}"
            full_path = os.path.join(folder_path, new_filename)

        # 保存图片
        cv2.imwrite(full_path, frame)
        saved_filenames.append(new_filename)
        # print(f"Saved image as: {new_filename}")
    return saved_filenames

def get_screen_size(screen_num):
    screens = []

    def callback(monitor, dc, rect, data):
        info = {}
        info['left'] = rect[0]
        info['top'] = rect[1]
        info['right'] = rect[2]
        info['bottom'] = rect[3]
        info['width'] = info['right'] - info['left']
        info['height'] = info['bottom'] - info['top']
        screens.append(info)
        return True

    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.c_long), ctypes.c_double)
    user32 = ctypes.windll.user32
    if not user32.EnumDisplayMonitors(0, 0, callback_type(callback), 0):
        raise RuntimeError("EnumDisplayMonitors failed")

    if screen_num < len(screens):
        screen = screens[screen_num]
        print(f"第{screen_num + 1}个屏幕的分辨率为: {screen['width']} x {screen['height']}")
        return screen['width'], screen['height']

    else:
        raise ValueError(f"Screen number {screen_num} is out of range.")


def get_image_files(folder_path, extensions=['.png', '.jpg', '.jpeg']):
    """
    获取指定文件夹中所有符合指定扩展名的图片文件路径。

    参数:
    folder_path : 文件夹路径
    extensions : 文件扩展名列表，默认为 ['.png', '.jpg', '.jpeg']

    返回:
    图片文件路径列表
    """
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    return image_files


def release_cv_window():
    cv2.destroyAllWindows()


def create_gray_pattern(w, h):
    # Python implementation of MATLAB's createGrayPattern ,最后的格雷码是四维 (n, h, w, 3)
    nbits = np.ceil(np.log2([w, h])).astype(int)  # # of bits for vertical/horizontal patterns
    offset = (2 ** nbits - [w, h]) // 2  # offset the binary pattern to be symmetric

    # coordinates to binary code
    c, r = np.meshgrid(np.arange(w), np.arange(h))
    bin_pattern = [np.unpackbits((c + offset[0])[..., None].view(np.uint8), axis=-1, bitorder='little', count=nbits[0])[..., ::-1],
                   np.unpackbits((r + offset[1])[..., None].view(np.uint8), axis=-1, bitorder='little', count=nbits[1])[..., ::-1]]

    # binary pattern to gray pattern
    gray_pattern = copy.deepcopy(bin_pattern)
    for n in range(len(bin_pattern)):
        for i in range(1, bin_pattern[n].shape[-1]):
            gray_pattern[n][:, :, i] = np.bitwise_xor(bin_pattern[n][:, :, i - 1], bin_pattern[n][:, :, i])

    # allPatterns also contains complementary patterns and all 0/1 patterns
    prj_patterns = np.zeros((h, w, 2 * sum(nbits) + 2), dtype=np.uint8)
    prj_patterns[:, :, 0] = 1  # All ones pattern

    # Vertical
    for i in range(gray_pattern[0].shape[-1]):
        prj_patterns[:, :, 2 * i + 2] = gray_pattern[0][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 3] = np.logical_not(gray_pattern[0][:, :, i]).astype(np.uint8)

    # Horizontal
    for i in range(gray_pattern[1].shape[-1]):
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 2] = gray_pattern[1][:, :, i].astype(np.uint8)
        prj_patterns[:, :, 2 * i + 2 * nbits[0] + 3] = np.logical_not(gray_pattern[1][:, :, i]).astype(np.uint8)

    prj_patterns *= 255

    # to RGB image
    # prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (0, 1, 3, 2))  # to (h, w, c, n)
    prj_patterns = np.transpose(np.tile(prj_patterns[..., None], (1, 1, 3)), (2, 0, 1, 3))  # to (n, h, w, c)

    return prj_patterns

