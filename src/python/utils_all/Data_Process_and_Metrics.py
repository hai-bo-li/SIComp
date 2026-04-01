import pandas as pd
from tqdm import tqdm
import lpips
from python.utils_all.differential_color_function import *
from .utils import *
# from Projector_Camera import compute_mask_from_paths, save_images
from .capture_utils import *

def save_metrics_to_excel(results_list, excel_path='results.xlsx', mode='overwrite'):
    """
    将评估指标保存到 Excel 文件中

    :param results_list: List[Dict]，每个字典包含模型名称、数据集名称及其指标
    :param excel_path: 要保存的 Excel 文件路径
    :param mode: 保存模式，'overwrite' 为覆盖写入，'append' 为追加写入（自动去重）
    """
    df_new = pd.DataFrame([results_list])

    if mode == 'overwrite' or not os.path.exists(excel_path):
        df_new.to_excel(excel_path, index=False)
        print(f"[写入] 结果已保存至 {excel_path}")
    elif mode == 'append':
        df_existing = pd.read_excel(excel_path)
        # 合并并去重（按Model和Dataset两列去重）
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all.drop_duplicates(subset=['model_name', 'data_name'], keep='last', inplace=True)
        df_all.to_excel(excel_path, index=False)
        print(f"[追加] 结果已追加保存至 {excel_path}")
    else:
        raise ValueError("mode 只能是 'overwrite' 或 'append'")


def join_path(*path_components):
    """  
    Join multiple path components into a single path using the appropriate separator for the operating system.  

    Args:  
        *path_components: Variable number of path components to be joined.  

    Returns:  
        str: Joined path string.  

    """
    return os.path.join(*path_components)


def create_mask_from_polygon(image_path, scale_factor=2.0):
    points = []  # List to store points
    drawing = False

    def draw_polygon(event, x, y, flags, param):
        nonlocal points, img, scale_factor, drawing
        x, y = int(x / scale_factor), int(y / scale_factor)
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if len(points) > 0:
                cv2.line(img, points[-1], (x, y), (0, 255, 0), 2)
            points.append((x, y))
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found at {image_path}")
        return
    img_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    cv2.setMouseCallback('image', draw_polygon)
    cv2.setWindowTitle('image', 'Draw by clicking with the mouse, press "m" to finish drawing, and press ESC to exit.')
    while True:
        cv2.imshow('image', cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR))
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            if len(points) > 2:
                cv2.fillPoly(img_mask, [np.array(points, dtype=np.int32)], (255))
                cv2.imshow('mask', img_mask)
            points = []
            drawing = False
    cv2.destroyAllWindows()
    # Get the parent directory two levels up from image_path
    mask_dir = join_path(os.path.split(os.path.split(os.path.split(image_path)[0])[0])[0], 'mask')
    # mask_dir = join_path(os.path.dirname(image_path), 'mask')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    mask_path = join_path(mask_dir, 'mask.png')
    cv2.imwrite(mask_path, img_mask)
    print(f"Mask saved to {mask_path}")
    return img_mask


def calculate_image_metrics(pred, target):
    if pred.max() > 1.0 or target.max() > 1.0:
        pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float() / 255
        target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float() / 255
    else:
        pred = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
        target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()
    """
    计算给定预测和目标图像之间的MSE, RMSE, PSNR和SSIM。
    :param pred: 预测的图像批次，tensor，shape为[batch_size, channels, height, width]
    :param target: 真实的图像批次，tensor，shape为[batch_size, channels, height, width]
    :return: mse, RMSE, PSNR, SSIM的值
    """
    pred = pred.to(device)
    target = target.to(device)
    # mse Loss
    # 注意，必须保证所有的函数都放置到了cuda上面
    # 同时.item()方法会把cuda上面的值移动到CPU上面
    #l1
    mae = l1_fun(pred, target).item()
    #l2
    mse = l2_fun(pred, target).item()
    # RMSE calculation
    rmse = math.sqrt(mse * 3)  # 乘以3因为是3个通道的平均
    # PSNR calculation
    if mse == 0:
        psnr = float('inf')  # 避免除以零的错误
    else:
        psnr = 10 * math.log10(1 / mse)
    # ssim calculation
    ssim = ssim_fun(pred, target).item()
    # deltaE计算
    xl_batch = rgb2lab_diff(pred, device)
    yl_batch = rgb2lab_diff(target, device)
    diff_map = ciede2000_diff(xl_batch, yl_batch, device)
    diff_map = diff_map.mean().item()
    # Lpips计算
    lpips = loss_fn_lpips(pred, target)
    lpips = lpips.mean().item()  # 取平均以得到标量损失
    return mae, rmse, psnr, ssim, diff_map, lpips


def create_desire_mask(image, rect):
    # 获取输入图像的高度和宽度
    height, width = image.shape[:2]
    # 创建与输入图像大小相同的全零图像
    mask = np.zeros((height, width), dtype=np.uint8)
    # 提取矩形的左上角坐标和宽度、高度
    x, y, w, h = rect
    # 计算矩形的右下角坐标
    x2, y2 = x + w, y + h
    # 将矩形区域内的像素值设为1
    mask[y:y2, x:x2] = 1
    return mask


def maximalRectangle(matrix, aspectRatio=1.0):
    if not matrix.size:
        return 0, None
    # 用于记录找到的最大矩形面积
    max_area = 0
    # 用于储存最大矩形的位置与尺寸,是一个tuple
    best_rect = (0, 0, 0, 0)  # (x, y, width, height)
    rows, cols = matrix.shape
    height = [0] * cols
    left = [0] * cols
    right = [cols] * cols
    for i in range(rows):
        cur_left, cur_right = 0, cols
        for j in range(cols):
            if matrix[i][j] == 1:
                height[j] += 1
                left[j] = max(left[j], cur_left)
            else:
                height[j] = 0
                left[j] = 0
                cur_left = j + 1
        for j in range(cols - 1, -1, -1):
            if matrix[i][j] == 1:
                right[j] = min(right[j], cur_right)
            else:
                right[j] = cols
                cur_right = j
        for j in range(cols):
            width = right[j] - left[j]
            ideal_height = width / aspectRatio
            effective_height = min(height[j], int(ideal_height))
            # 当 aspectRatio 为 1 时，确保宽度和高度相等
            if aspectRatio == 1.0:
                effective_height = min(width, height[j])
                width = effective_height
            current_area = effective_height * width
            if current_area > max_area:
                max_area = current_area
                best_rect = (left[j], i - effective_height + 1, width, effective_height)
    return max_area, best_rect


def calculate_bbox_from_mask(mask_image):
    # Convert mask_image to binary format (0 or 255)
    mask_binary = (mask_image > 0).astype(np.uint8)
    # Find contours of the mask
    contours, _ = cv.findContours(mask_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Find the minimum bounding rectangle of the largest contour
    bbox = cv.boundingRect(max(contours, key=cv.contourArea))
    return bbox


def place_image_in_rectangle_opencv(image, mask, rectangle):
    # 获取矩形的尺寸和位置
    x, y, width, height = rectangle
    # 根据矩形尺寸调整图像大小
    resized_image = cv.resize(image, (width, height))
    # 创建一个新的图像，大小与mask相同，初始化为全黑或其他背景
    # mask 应该是一个二维数组，这里我们创建一个与其同样大小的三通道图像
    new_image = np.zeros_like(mask)
    if len(mask.shape) == 2:  # 如果mask是单通道，转换为三通道
        new_image = cv.cvtColor(new_image, cv.COLOR_GRAY2BGR)
    # 将调整后的图像放置到新图像的指定位置
    new_image[y:y + height, x:x + width] = resized_image
    # cv.imshow('Processed Image', new_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return new_image


def get_mask(data_path, dataset):
    """
    根据提供的 data_path 和 dataset 构造 mask.png 文件的完整路径，并使用 OpenCV 读取它。

    参数：
        data_path (str): 数据的基本路径。
        dataset (str): 特定数据集的名称。

    返回：
        im_mask (numpy.ndarray 或 None): 读取的图像。如果文件不存在或读取失败，则返回 None。
    """
    # 构造 mask.png 的路径
    mask_path = join_path(data_path, dataset, 'cam', 'mask', 'mask.png')
    print("Mask path:", mask_path)

    # 检查路径并读取图像
    if os.path.exists(mask_path):
        # 使用 OpenCV 加载图像
        im_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if im_mask is not None:
            print("成功读取遮罩文件。")
            return im_mask
        else:
            print("读取遮罩文件时出错。")
            return None
    else:
        print("遮罩文件不存在。")
        return None


def ensure_directories(*dirs):
    """
    安全地创建目录，避免可能的栈溢出问题

    Args:
        *dirs: 需要确保存在的目录路径
    """
    for directory in dirs:
        # 验证路径合法性
        if directory and isinstance(directory, str):
            try:
                path = os.path.abspath(directory)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    print(f"已创建目录: {path}")
            except Exception as e:
                print(f"创建目录 {directory} 时出错: {e}")
        else:
            print(f"无效的目录路径: {directory}")

def crop_image(image, bbox):
    """
    根据给定的边界框裁切图像.

    :param image: 输入图像 (numpy array)
    :param bbox: 边界框 (x1, y1, w1, h1)
    :return: 裁切后的图像 (numpy array)
    """
    x1, y1, w1, h1 = bbox
    # 计算裁切区域
    crop = image[y1:y1 + h1, x1:x1 + w1]
    return crop


def visualize_images(raw_cmp_img, full_desired_img):
    """
    Visualize two images using OpenCV.

    Parameters:
    -----------
    raw_cmp_img : numpy.ndarray
        The first image (raw comparison image).
    full_desired_img : numpy.ndarray
        The second image (ground truth image).

    Returns:
    --------
    None
    """
    # Check if images are loaded successfully
    if raw_cmp_img is None:
        raise ValueError("Raw comparison image is not valid.")

    if full_desired_img is None:
        raise ValueError("Ground truth image is not valid.")

        # Visualize images
    cv2.imshow('Raw Comparison Image', raw_cmp_img)
    cv2.imshow('Ground Truth Image', full_desired_img)

    # Wait for a key press and then close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_images_with_path(raw_cmp_path, full_desired_img_path):
    """
    Read and visualize images using OpenCV from two different paths.

    Parameters:
    -----------
    raw_cmp_path : str
        Path to the first image (raw comparison image)
    full_desired_img_path : str
        Path to the second image (ground truth image)

    Returns:
    --------
    tuple
        A tuple containing the two loaded images
    """
    # Check if files exist
    if not os.path.exists(raw_cmp_path):
        raise FileNotFoundError(f"Raw comparison image not found: {raw_cmp_path}")

    if not os.path.exists(full_desired_img_path):
        raise FileNotFoundError(f"Ground truth image not found: {full_desired_img_path}")

        # Read images
    raw_cmp_img = cv2.imread(raw_cmp_path)
    full_desired_img = cv2.imread(full_desired_img_path)

    # Check if images are loaded successfully
    if raw_cmp_img is None:
        raise ValueError(f"Failed to read raw comparison image: {raw_cmp_path}")

    if full_desired_img is None:
        raise ValueError(f"Failed to read ground truth image: {full_desired_img_path}")

        # Visualize images
    cv2.imshow('Raw Comparison Image', raw_cmp_img)
    cv2.imshow('Ground Truth Image', full_desired_img)

    # Wait for a key press and then close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return raw_cmp_img, full_desired_img


def apply_mask_and_crop_images_actualMetric(data_path, data_list, target_size=(256, 256), metrics=True, im_mask_bool=False):
    use_im_mask = im_mask_bool
    all_results = []
    for dataset in data_list:
        dataset = join_path('setups', dataset)
        ref_min_path = join_path(data_path, dataset, 'cam', 'raw', 'ref', 'img_0001.png')
        ref_max_path = join_path(data_path, dataset, 'cam', 'raw', 'ref', 'img_0125.png')
        mask_path = join_path(data_path, dataset, 'cam', 'mask')
        # mask path cam/mask/mask.png
        if use_im_mask:
            im_mask = get_mask(data_path, dataset)
        else:
            # 后续都取消手动绘制的过程，因为这样每一次绘制的mask都会存在微小差异
            # im_mask = create_mask_from_polygon(ref_max_path, scale_factor=2.0)
            im_mask, _ = compute_mask_from_paths(ref_min_path,ref_max_path)
            save_images(im_mask, mask_path, filename='mask', overwrite=True)
        for phase in ['train', 'test']:
            # 用于保存对于设置train\test  crop地址
            raw_dir = join_path(data_path, dataset, 'cam', 'raw', phase)
            crop_dir = join_path(data_path, dataset, 'cam', 'crop', phase)
            # 用于保存crop 表面ref的地址
            crop_ref_dir = join_path(data_path, dataset, 'cam', 'crop', 'ref')
            # 如果需要采用不同的mask，需要修改这里的mask地址
            # 用于保存mask的位置
            mask_dir = join_path(data_path, dataset, 'cam', 'mask')
            # 保存处理后的真实数据地址，也就是mask后的真实数据
            raw_desired_masked_cmp_dir = join_path(data_path, dataset, 'cam', 'desire', 'masked', phase)
            ensure_directories(crop_dir, mask_dir, crop_ref_dir, raw_desired_masked_cmp_dir)
            # 表面背景的保存图像位置
            ref_dir = join_path(data_path, dataset, 'cam', 'raw', 'ref')
            # 用于计算mask的图片125与1号图片
            # img1_path = join_path(ref_dir, 'img_0125.png')
            img2_path = join_path(ref_dir, 'img_0001.png')
            img2 = cv.imread(img2_path)
            bbox = calculate_bbox_from_mask(im_mask)
            im_mask = im_mask > 0
            x1, y1, w1, h1 = bbox
            mask_cropped = im_mask.astype(np.uint8)[y1:y1 + h1, x1:x1 + w1]
            im_mask = im_mask.astype(np.uint8)
            # 返回最大内接矩形，需要注意的是，这里的输入的mask_cropped一定是0，255格式！
            _, best_rect = maximalRectangle(mask_cropped, aspectRatio=1)
            # full_best_rect是desired_image的mask，用于后面的真实数据比较
            _, full_best_rect = maximalRectangle(im_mask, aspectRatio=1)
            # desired_mask，输入的img1，也就是根据img1的图片大小，生成一个全0的二维矩阵，然后根据full_best_rect也就最大内接矩形来将其内部变为255白色，以便mask
            desired_mask = create_desire_mask(img2, full_best_rect)
            if not metrics:
                if best_rect:
                    x, y, width, height = best_rect
                    # 在原始掩码上绘制红色矩形框
                    colored_mask = cv.cvtColor(mask_cropped * 255, cv.COLOR_GRAY2BGR)  # 转换为 BGR 彩色图像
                    cv.rectangle(colored_mask, (x, y), (x + width, y + height), (0, 0, 255), 1)  # 红色框，2 像素宽
                    # 保存带有红框的图像到文件
                    output_folder = mask_dir
                    filename = 'mask_with_rectangle.png'
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    cv.imwrite(join_path(output_folder, filename), colored_mask)
                else:
                    print("No rectangle found")
        crop_ref_dir = join_path(data_path, dataset, 'cam', 'crop', 'ref')
        raw_desired_masked_cmp_dir = join_path(data_path, dataset, 'cam', 'desire', 'masked', 'test')
        ensure_directories(mask_dir, crop_ref_dir, raw_desired_masked_cmp_dir)
        if not metrics:
            for phase in ['train', 'test']:
                # 用于保存对于设置train\test  crop地址
                raw_dir = join_path(data_path, dataset, 'cam', 'raw', phase)
                crop_dir = join_path(data_path, dataset, 'cam', 'crop', phase)
                ensure_directories(raw_dir, crop_dir)
                file_count = 500 if phase == 'train' else 200
                for i in tqdm(range(1, file_count + 1)):
                    img_name = f'img_{i:04d}.png'
                    img_path = join_path(raw_dir, img_name)
                    if os.path.exists(img_path):
                        img = cv.imread(img_path)
                        if img is not None and im_mask is not None:
                            img_cropped = img[y1:y1 + h1, x1:x1 + w1]
                            masked_img = cv.bitwise_and(img_cropped, img_cropped, mask=mask_cropped)
                            resized_img = cv.resize(masked_img, target_size)
                            cropped_img_path = join_path(crop_dir, img_name)
                            # 这里有一个坑，就是保存的时候不能是BGR格式
                            cv.imwrite(cropped_img_path, resized_img)
                print(f'{phase} masked and resized image saved to {crop_dir}')
        # ref process
        raw_ref_dir = join_path(data_path, dataset, 'cam', 'raw', 'ref')
        if not metrics:
            for i in range(1, 127):
                img_name = f'img_{i:04d}.png'
                img_path = join_path(raw_ref_dir, img_name)
                if os.path.exists(img_path):
                    img = cv.imread(img_path)
                    if img is not None and im_mask is not None:
                        img_cropped = img[y1:y1 + h1, x1:x1 + w1]
                        masked_img = cv.bitwise_and(img_cropped, img_cropped, mask=mask_cropped)
                        resized_img = cv.resize(masked_img, target_size)
                        cropped_img_path = join_path(crop_ref_dir, img_name)
                        # 使用opencv读取，然后也使用opencv保存
                        cv.imwrite(cropped_img_path, resized_img)
            print(f'Reference image processed and saved to {crop_ref_dir}')
        # desire地址，这个desire是需要根据最大内接矩形生成的
        GT = join_path(data_path, 'test')
        # crop desire保存地址
        crop_desire_dir = join_path(data_path, dataset, 'cam', 'crop', 'desire')
        desired_GT_dir = join_path(data_path, dataset, 'cam', 'desire', "test")
        # Create the crop_desire_dir if it doesn't exist
        ensure_directories(crop_desire_dir, desired_GT_dir)
        real_mask_metrics = {
            'mae': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0,
            'diff_map': 0, 'lpips': 0  # 添加 diff_map 和 lpips
        }
        count = 0
        # 这里是保存desired_image用于真实指标计算
        # 获取所有 GT 文件夹下的图片路径
        from glob import glob
        image_paths = sorted(glob(os.path.join(GT, 'img_*.png')))  # 或者直接 os.listdir(GT)，但 glob 可筛选格式
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            # 读取GT图片
            img = cv.imread(img_path)
            # 生成真实数据mask，然后用于计算指标，存在一个问题就是mask是根据矩形生成的，然后如果光流扭不正就会掩盖这个错误
            # 这是保存的desired_image，根据GT来保存，先预处理数据
            desired_mask = np.uint8(desired_mask) * 255
            full_desired_GT = place_image_in_rectangle_opencv(img, desired_mask, rectangle=full_best_rect)
            # 这是保存的desired_image，根据GT来保存
            full_desired_img_path = join_path(desired_GT_dir, img_name)
            cv.imwrite(full_desired_img_path, full_desired_GT)

            if os.path.exists(img_path):
                img = cv.imread(img_path)
                if img is not None and im_mask is not None:
                    # _, Largest_Inscribed_Rectangle = maximalRectangle(mask_cropped, aspectRatio=1)
                    desire_GT = place_image_in_rectangle_opencv(img, mask_cropped, rectangle=best_rect)
                    resized_img = cv.resize(desire_GT, target_size)
                    cropped_img_path = join_path(crop_desire_dir, img_name)
                    # 这里保存的是用于光流的desired图片
                    cv.imwrite(cropped_img_path, resized_img)
        if metrics:
            print('actual comparison started.\n')
            # Define the list of compensation folders to process
            raw_cmp_root = join_path(data_path, dataset, 'prj', 'cmp')
            cmp_folders = [f for f in os.listdir(raw_cmp_root) if os.path.isdir(join_path(raw_cmp_root, f))]
            # Process each compensation folder
            for cmp_folder in cmp_folders:
                print(f"Processing {cmp_folder}...")
                # Define paths for current compensation folder
                raw_cmp_dir = join_path(data_path, dataset, 'cam', 'raw', cmp_folder)

                # Create directory for masked compensations based on folder name
                setup_raw_desired_masked_cmp_dir = join_path(raw_desired_masked_cmp_dir, f"masked_{cmp_folder}")
                os.makedirs(setup_raw_desired_masked_cmp_dir, exist_ok=True)

                # Reset metrics for this folder
                if metrics:
                    real_mask_metrics = {
                        'model_name':f'{cmp_folder}',
                        'data_name':f'{dataset}',
                        'rmse': 0,
                        'psnr': 0,
                        'ssim': 0,
                        'diff_map': 0,
                        'lpips': 0
                    }
                    real_unmask_metrics = {
                        'model_name': f'{cmp_folder}',
                        'data_name': f'{dataset}',
                        'rmse': 0,
                        'psnr': 0,
                        'ssim': 0,
                        'diff_map': 0,
                        'lpips': 0
                    }
                    avg_real_mask_metrics = {
                        'model_name': f'{cmp_folder}',
                        'data_name': f'{dataset}',
                        'rmse': 0,
                        'psnr': 0,
                        'ssim': 0,
                        'diff_map': 0,
                        'lpips': 0
                    }
                    avg_real_unmask_metrics = {
                        'model_name': f'{cmp_folder}',
                        'data_name': f'{dataset}',
                        'rmse': 0,
                        'psnr': 0,
                        'ssim': 0,
                        'diff_map': 0,
                        'lpips': 0
                    }
                    count = 0

                    # Process images in the current folder
                if os.path.exists(raw_cmp_dir) and metrics:
                    for img_name in os.listdir(crop_desire_dir):
                        if img_name.endswith('.png') or img_name.endswith('.jpg'):
                            # 处理真实补偿，为其加上mask，使其与desired_image大小格式相同
                            raw_cmp_path = join_path(raw_cmp_dir, img_name)
                            full_desired_img_path = join_path(desired_GT_dir,img_name)
                            # 可视化对比图片，确认正确的读取
                            # read_and_visualize_image(raw_cmp_path,full_desired_img_path)
                            # Check if the image exists in this compensation folder
                            if not os.path.exists(raw_cmp_path):
                                print(f"Warning: {img_name} not found in {cmp_folder}, skipping.")
                                continue

                            raw_img_cmp = cv.imread(raw_cmp_path)
                            full_desired_img = cv.imread(full_desired_img_path)
                            masked_raw_img_cmp = cv.bitwise_and(raw_img_cmp, raw_img_cmp, mask=desired_mask)
                            crop_full_desired_img = crop_image(full_desired_img,bbox)
                            crop_masked_raw_img_cmp = crop_image(masked_raw_img_cmp,bbox)
                            crop_raw_img_cmp = crop_image(raw_img_cmp,bbox)
                            # visualize_images(crop_masked_raw_img_cmp, crop_full_desired_img)
                            # 这是保存的 相机拍摄的mask后的补偿图像，用于与desired_image做计算
                            setup_raw_desired_masked_cmp_path = join_path(setup_raw_desired_masked_cmp_dir, img_name)
                            cv.imwrite(setup_raw_desired_masked_cmp_path, masked_raw_img_cmp)

                            if metrics:
                                _, rmse, psnr, ssim, diff_map, lpips = calculate_image_metrics(crop_masked_raw_img_cmp, crop_full_desired_img)
                                real_mask_metrics['rmse'] += rmse
                                real_mask_metrics['psnr'] += psnr
                                real_mask_metrics['ssim'] += ssim
                                real_mask_metrics['diff_map'] += diff_map
                                real_mask_metrics['lpips'] += lpips

                                _, u_rmse, u_psnr, u_ssim, u_diff_map, u_lpips = calculate_image_metrics(crop_raw_img_cmp,
                                                                                                         crop_full_desired_img)
                                real_unmask_metrics['rmse'] += u_rmse
                                real_unmask_metrics['psnr'] += u_psnr
                                real_unmask_metrics['ssim'] += u_ssim
                                real_unmask_metrics['diff_map'] += u_diff_map
                                real_unmask_metrics['lpips'] += u_lpips
                                count += 1
                else:
                    print(f"Warning: Directory {raw_cmp_dir} does not exist, skipping.")
                    continue

                    # Print results for current folder
                if metrics and count > 0:
                    print(f"Real_Mask_Metrics for {cmp_folder}:")
                    avg_rmse = real_mask_metrics['rmse'] / count
                    avg_real_mask_metrics['rmse'] = avg_rmse
                    avg_psnr = real_mask_metrics['psnr'] / count
                    avg_real_mask_metrics['psnr']=avg_psnr
                    avg_ssim = real_mask_metrics['ssim'] / count
                    avg_real_mask_metrics['ssim'] = avg_ssim
                    avg_diff_map = real_mask_metrics['diff_map'] / count
                    avg_real_mask_metrics['diff_map'] = avg_diff_map
                    avg_lpips = real_mask_metrics['lpips'] / count
                    avg_real_mask_metrics['lpips'] = avg_lpips
                    print(f' Average RMSE: {avg_rmse:.4f} Average PSNR: {avg_psnr:.4f} Average SSIM: {avg_ssim:.4f} Average Diff: {avg_diff_map:.4f} Average LPIPS: {avg_lpips:.4f}')
                    print(f"Real_UnMask_Metrics for {cmp_folder}:")
                    uavg_rmse = real_unmask_metrics['rmse'] / count
                    avg_real_unmask_metrics['rmse'] = uavg_rmse
                    uavg_psnr = real_unmask_metrics['psnr'] / count
                    avg_real_unmask_metrics['psnr'] = uavg_psnr
                    uavg_ssim = real_unmask_metrics['ssim'] / count
                    avg_real_unmask_metrics['ssim'] = uavg_ssim
                    uavg_diff_map = real_unmask_metrics['diff_map'] / count
                    avg_real_unmask_metrics['diff_map'] = uavg_diff_map
                    uavg_lpips = real_unmask_metrics['lpips'] / count
                    avg_real_unmask_metrics['lpips'] = uavg_lpips
                    mask_excel_file_path = join_path(raw_cmp_path,'mask_actual_metrics.xlsx')
                    umask_excek_file_path = join_path(raw_cmp_path,'umask_actual_metrics.xlsx')
                    save_metrics_to_excel(avg_real_mask_metrics, excel_path=mask_excel_file_path, mode='append')
                    save_metrics_to_excel(avg_real_unmask_metrics, excel_path=umask_excek_file_path, mode='append')
                    print(f'Average PSNR: {uavg_psnr:.4f} Average RMSE: {uavg_rmse:.4f}  Average SSIM: {uavg_ssim:.4f} Average Diff: {uavg_diff_map:.4f} Average LPIPS: {uavg_lpips:.4f}\n')
                    # print(f'{uavg_rmse:.4f} {uavg_psnr:.4f} {uavg_ssim:.4f} {uavg_diff_map:.4f} {uavg_lpips:.4f}\n')
                elif metrics:
                    print(f"No images processed for {cmp_folder}.")
        print('Crop desire image processed and saved to', crop_desire_dir)
        print('full_GT_desired_images processd and saved to', desired_GT_dir)
        print('All processing done.')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    ssim_fun = pytorch_ssim.SSIM().to(device)

    dataset_root = r'xxxx'
    data_name = [r"xxxx"]
    apply_mask_and_crop_images_actualMetric(dataset_root, data_name, target_size=(600, 600), metrics=False, im_mask_bool=True)
