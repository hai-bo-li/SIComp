import csv
import os
import cv2
from tqdm import tqdm
import lpips
from differential_color_function import *
import datetime
import warnings
from .utils import *
from .capture_utils import *
from Data_Process_and_Metrics import save_metrics_to_excel

def append_metrics_to_csv(log_path, dataset_name, model_name, mask_metrics, unmask_metrics, count):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_path)

    with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if the file does not exist
        if not file_exists:
            writer.writerow([
                "timestamp", "dataset", "model", "count",
                "MAE_mask", "RMSE_mask", "PSNR_mask", "SSIM_mask", "Diff_mask", "LPIPS_mask",
                "MAE_unmask", "RMSE_unmask", "PSNR_unmask", "SSIM_unmask", "Diff_unmask", "LPIPS_unmask"
            ])

        # Write metric values
        writer.writerow([
            timestamp, dataset_name, model_name, count,
            mask_metrics['mae'], mask_metrics['rmse'], mask_metrics['psnr'],
            mask_metrics['ssim'], mask_metrics['diff_map'], mask_metrics['lpips'],
            unmask_metrics['mae'], unmask_metrics['rmse'], unmask_metrics['psnr'],
            unmask_metrics['ssim'], unmask_metrics['diff_map'], unmask_metrics['lpips']
        ])


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
    Compute MSE, RMSE, PSNR, and SSIM between the predicted image and the target image.

    :param pred: Predicted image batch, tensor with shape [batch_size, channels, height, width]
    :param target: Ground-truth image batch, tensor with shape [batch_size, channels, height, width]
    :return: Values of MSE, RMSE, PSNR, and SSIM
    """
    pred = pred.to(device)
    target = target.to(device)
    #l1
    mae = l1_fun(pred, target).item()
    #l2
    mse = l2_fun(pred, target).item()
    # RMSE calculation
    rmse = math.sqrt(mse)
    # PSNR calculation
    if mse == 0:
        psnr = float('inf')  # Avoid division-by-zero errors
    else:
        psnr = 10 * math.log10(1 / mse)
    # ssim calculation
    ssim = ssim_fun(pred, target).item()
    # DeltaE computation
    xl_batch = rgb2lab_diff(pred, device)
    yl_batch = rgb2lab_diff(target, device)
    diff_map = ciede2000_diff(xl_batch, yl_batch, device)
    diff_map = diff_map.mean().item()
    # LPIPS computation
    lpips = loss_fn_lpips(pred, target)
    lpips = lpips.mean().item()  # Take the mean to obtain a scalar loss
    return mae, rmse, psnr, ssim, diff_map, lpips


def create_desire_mask(image, rect):
    # Get the height and width of the input image
    height, width = image.shape[:2]
    # Create an all-zero image with the same size as the input image
    mask = np.zeros((height, width), dtype=np.uint8)
    # Extract the top-left coordinates, width, and height of the rectangle
    x, y, w, h = rect
    # Compute the bottom-right coordinates of the rectangle
    x2, y2 = x + w, y + h
    # Set pixel values inside the rectangle region to 1
    mask[y:y2, x:x2] = 1
    return mask


def maximalRectangle(matrix, aspectRatio=1.0):
    if not matrix.size:
        return 0, None
    # Used to record the maximum rectangle area found
    max_area = 0
    # Used to store the position and size of the maximum rectangle as a tuple
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
            # When aspectRatio is 1, ensure equal width and height
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
    # Get the size and position of the rectangle
    x, y, width, height = rectangle
    # Resize the image according to the rectangle size
    resized_image = cv.resize(image, (width, height))
    # Create a new image with the same size as the mask and initialize it as black or another background
    # The mask should be a 2D array, so here we create a three-channel image with the same size
    new_image = np.zeros_like(mask)
    if len(mask.shape) == 2:  # If the mask is single-channel, convert it to three channels
        new_image = cv.cvtColor(new_image, cv.COLOR_GRAY2BGR)
    # Place the resized image into the specified location in the new image
    new_image[y:y + height, x:x + width] = resized_image
    # cv.imshow('Processed Image', new_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return new_image


def get_mask(data_path, dataset):
    """
    Build the full path to `mask.png` from `data_path` and `dataset`, then read it using OpenCV.

    Args:
        data_path (str): Base path of the data.
        dataset (str): Name of the specific dataset.

    Returns:
        im_mask (numpy.ndarray or None): Loaded image. Returns None if the file does not exist or cannot be read.
    """
    # Build the path to mask.png
    mask_path = join_path(data_path, dataset, 'cam', 'raw', 'mask', 'mask.png')
    print("Mask path:", mask_path)

    # Check the path and read the image
    if os.path.exists(mask_path):
        # Load the image using OpenCV
        im_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if im_mask is not None:
            print("Successfully loaded the mask file.")
            return im_mask
        else:
            print("Error reading the mask file.")
            return None
    else:
        print("The mask file does not exist.")
        return None


def ensure_directories(*dirs):
    """
    Safely create directories to avoid possible stack overflow issues.

    Args:
        *dirs: Directory paths that need to be ensured to exist
    """
    for directory in dirs:
        # Validate whether the path is legal
        if directory and isinstance(directory, str):
            try:
                path = os.path.abspath(directory)
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    print(f"Created directory: {path}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
        else:
            print(f"Invalid directory path: {directory}")


def crop_image(image, bbox):
    """
    Crop an image according to the given bounding box.

    :param image: Input image (numpy array)
    :param bbox: Bounding box (x1, y1, w1, h1)
    :return: Cropped image (numpy array)
    """
    x1, y1, w1, h1 = bbox
    # Compute the crop region
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


def apply_mask_and_crop_images_actualMetric(data_path, data_list, target_size=(256, 256), metrics=False, im_mask_bool=False):
    use_im_mask = im_mask_bool
    all_results = []
    for dataset in data_list:
        # dataset = join_path('setups', dataset)
        ref_min_path = join_path(data_path, dataset, 'cam', 'raw', 'ref', 'img_0001.png')
        ref_max_path = join_path(data_path, dataset, 'cam', 'raw', 'ref', 'img_0125.png')
        mask_path = join_path(data_path, dataset, 'cam', 'mask')
        # mask path cam/mask/mask.png
        if use_im_mask:
            im_mask = get_mask(data_path, dataset)
        else:
            # Manual mask drawing is disabled from this point on, because each manual draw introduces slight differences
            # im_mask = create_mask_from_polygon(ref_max_path, scale_factor=2.0)
            im_mask, _ = compute_mask_from_paths(ref_min_path, ref_max_path)
            save_images(im_mask, mask_path, filename='mask', overwrite=True)
        for phase in ['train', 'test']:
            # Save paths for cropped train/test images under the current setup
            raw_dir = join_path(data_path, dataset, 'cam', 'raw', phase)
            crop_dir = join_path(data_path, dataset, 'cam', 'crop', phase)
            # Save path for cropped surface reference images
            crop_ref_dir = join_path(data_path, dataset, 'cam', 'crop', 'ref')
            # If a different mask is needed, modify the mask path here
            # Save path for the mask
            mask_dir = join_path(data_path, dataset, 'cam', 'mask')
            # Save path for processed real data, i.e., masked real data
            raw_desired_masked_cmp_dir = join_path(data_path, dataset, 'cam', 'desire', 'masked', phase)
            ensure_directories(crop_dir, mask_dir, crop_ref_dir, raw_desired_masked_cmp_dir)
            # Save path for background surface images
            ref_dir = join_path(data_path, dataset, 'cam', 'raw', 'ref')
            # Images 125 and 1 are used to compute the mask
            img2_path = join_path(ref_dir, 'img_0001.png')
            img2 = cv.imread(img2_path)
            bbox = calculate_bbox_from_mask(im_mask)
            im_mask = im_mask > 0
            x1, y1, w1, h1 = bbox
            mask_cropped = im_mask.astype(np.uint8)[y1:y1 + h1, x1:x1 + w1]
            im_mask = im_mask.astype(np.uint8)
            # Return the largest inscribed rectangle; note that mask_cropped must be in 0/255 format
            _, best_rect = maximalRectangle(mask_cropped, aspectRatio=1)
            # full_best_rect is the mask of desired_image, used for real-data comparison later
            _, full_best_rect = maximalRectangle(im_mask, aspectRatio=1)
            # desired_mask is a full-zero 2D matrix generated from the size of img1.
            # Then the region inside full_best_rect, i.e. the largest inscribed rectangle, is set to white for masking.
            desired_mask = create_desire_mask(img2, full_best_rect)
            if not metrics:
                if best_rect:
                    x, y, width, height = best_rect
                    # Draw a red rectangle on the original mask
                    colored_mask = cv.cvtColor(mask_cropped * 255, cv.COLOR_GRAY2BGR)  # Convert to a BGR color image
                    cv.rectangle(colored_mask, (x, y), (x + width, y + height), (0, 0, 255), 1)  # Red rectangle, 1-pixel width
                    # Save the image with the red rectangle to a file
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
                # Save paths for cropped train/test images under the current setup
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
                            # There is a caveat here: the saved image should not be in BGR format
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
                        # Read using OpenCV and save using OpenCV as well
                        cv.imwrite(cropped_img_path, resized_img)
            print(f'Reference image processed and saved to {crop_ref_dir}')
        # Path for desire images; this desire image needs to be generated from the largest inscribed rectangle
        GT = join_path(data_path, 'test')
        # crop desire保存地址
        crop_desire_dir = join_path(data_path, dataset, 'cam', 'crop', 'desire')
        desired_GT_dir = join_path(data_path, dataset, 'cam', 'desire', "test")
        # Create the crop_desire_dir if it doesn't exist
        ensure_directories(crop_desire_dir, desired_GT_dir)
        count = 0
        # This is where desired_image is saved for real-metric computation
        # Get all image paths under the GT folder
        from glob import glob
        image_paths = sorted(glob(os.path.join(GT, 'img_*.png')))  # Or use os.listdir(GT) directly, but glob can filter by format
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            # Read the GT image
            img = cv.imread(img_path)
            # Generate the mask for real-data evaluation. One issue is that the mask is rectangle-based,
            # so if the optical flow is not well aligned, this error may be partially hidden.
            # Save desired_image generated from GT as a preprocessing step
            desired_mask = np.uint8(desired_mask) * 255
            full_desired_GT = place_image_in_rectangle_opencv(img, desired_mask, rectangle=full_best_rect)
            # Save desired_image generated from GT
            full_desired_img_path = join_path(desired_GT_dir, img_name)
            cv.imwrite(full_desired_img_path, full_desired_GT)

            if os.path.exists(img_path):
                img = cv.imread(img_path)
                if img is not None and im_mask is not None:
                    # _, Largest_Inscribed_Rectangle = maximalRectangle(mask_cropped, aspectRatio=1)
                    desire_GT = place_image_in_rectangle_opencv(img, mask_cropped, rectangle=best_rect)
                    resized_img = cv.resize(desire_GT, target_size)
                    cropped_img_path = join_path(crop_desire_dir, img_name)
                    # Save the desired image used for optical flow
                    cv.imwrite(cropped_img_path, resized_img)

        print('actual comparison started.\n')
        # Define the list of compensation folders to process
        raw_cmp_root = join_path(data_path, dataset, 'prj', 'cmp')
        cmp_folders = [f for f in os.listdir(raw_cmp_root) if os.path.isdir(join_path(raw_cmp_root, f))]
        # Process each compensation folder
        for cmp_folder in cmp_folders:
            print(f"Processing {cmp_folder}...")
            # Define paths for current compensation folder
            raw_cmp_dir = join_path(data_path, dataset, 'cam', 'raw', cmp_folder)


            setup_raw_desired_masked_cmp_dir = join_path(raw_desired_masked_cmp_dir, f"masked_{cmp_folder}")
            os.makedirs(setup_raw_desired_masked_cmp_dir, exist_ok=True)

            # Reset metrics for this folder
            # if metrics:
            real_mask_metrics = {
                'model_name': f'{cmp_folder}',
                'data_name': f'{dataset}',
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

            if os.path.exists(raw_cmp_dir) and metrics:
                for img_name in os.listdir(crop_desire_dir):
                    if img_name.endswith('.png') or img_name.endswith('.jpg'):
                        # Process real compensation by applying the mask so that it has the same size and format as desired_image
                        raw_cmp_path = join_path(raw_cmp_dir, img_name)
                        full_desired_img_path = join_path(desired_GT_dir, img_name)
                        # Visualize the comparison images to confirm correct loading
                        # read_and_visualize_image(raw_cmp_path,full_desired_img_path)
                        # Check if the image exists in this compensation folder
                        if not os.path.exists(raw_cmp_path):
                            print(f"Warning: {img_name} not found in {cmp_folder}, skipping.")
                            continue

                        raw_img_cmp = cv.imread(raw_cmp_path)
                        full_desired_img = cv.imread(full_desired_img_path)
                        masked_raw_img_cmp = cv.bitwise_and(raw_img_cmp, raw_img_cmp, mask=desired_mask)
                        crop_full_desired_img = crop_image(full_desired_img, bbox)
                        crop_masked_raw_img_cmp = crop_image(masked_raw_img_cmp, bbox)
                        crop_raw_img_cmp = crop_image(raw_img_cmp, bbox)
                        # visualize_images(crop_masked_raw_img_cmp, crop_full_desired_img)
                        # This is the saved masked compensation image captured by the camera, used for comparison with desired_image
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
                # Print results for the current folder
            if metrics and count > 0:
                print(f"Real_Mask_Metrics for {cmp_folder}:")
                avg_rmse = real_mask_metrics['rmse'] / count
                avg_real_mask_metrics['rmse'] = avg_rmse
                avg_psnr = real_mask_metrics['psnr'] / count
                avg_real_mask_metrics['psnr'] = avg_psnr
                avg_ssim = real_mask_metrics['ssim'] / count
                avg_real_mask_metrics['ssim'] = avg_ssim
                avg_diff_map = real_mask_metrics['diff_map'] / count
                avg_real_mask_metrics['diff_map'] = avg_diff_map
                avg_lpips = real_mask_metrics['lpips'] / count
                avg_real_mask_metrics['lpips'] = avg_lpips
                print(
                    f' Average RMSE: {avg_rmse:.4f} Average PSNR: {avg_psnr:.4f} Average SSIM: {avg_ssim:.4f} Average Diff: {avg_diff_map:.4f} Average LPIPS: {avg_lpips:.4f}')
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
                mask_excel_file_path = join_path(raw_cmp_root, 'mask_actual_metrics.xlsx')
                umask_excek_file_path = join_path(raw_cmp_root, 'umask_actual_metrics.xlsx')
                save_metrics_to_excel(avg_real_mask_metrics, excel_path=mask_excel_file_path, mode='append')
                save_metrics_to_excel(avg_real_unmask_metrics, excel_path=umask_excek_file_path, mode='append')
                print(
                    f'Average PSNR: {uavg_psnr:.4f} Average RMSE: {uavg_rmse:.4f}  Average SSIM: {uavg_ssim:.4f} Average Diff: {uavg_diff_map:.4f} Average LPIPS: {uavg_lpips:.4f}\n')
            elif metrics:
                print(f"No images processed for {cmp_folder}.")
        print('All processing done.')


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warp_str = os.environ.get("WARP_IM_SZ_STR", "1024 1024")
    warp_im_sz = tuple(map(int, warp_str.split()))
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    ssim_fun = pytorch_ssim.SSIM().to(device)
    # data_root = r"H:/test_projector_camera/DPCS_600_real"
    data_root = os.getenv("DATASET_ROOT","")
    data_list = get_linux_style_dataset_list(dataset_root=data_root, include_prefixes=[os.getenv("DATA_NAME","")])
    apply_mask_and_crop_images_actualMetric(data_root, data_list, warp_im_sz, metrics=True, im_mask_bool=True)
