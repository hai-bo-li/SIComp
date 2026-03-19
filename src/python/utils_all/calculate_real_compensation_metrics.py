import pandas as pd
from tqdm import tqdm
import lpips
from pytorch_fid import fid_score
from .capture_utils import *
# Assume custom utility modules are already available in the import path
from python.utils_all.differential_color_function import *
from .utils import *
from .pytorch_ssim import SSIM as pytorch_ssim_fun

def maximalRectangle(matrix, aspectRatio=1.0):
    if not matrix.size:
        return 0, None
    # Used to record the area of the found maximum rectangle
    max_area = 0
    # Used to store the position and size of the maximum rectangle, as a tuple
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
            # When aspectRatio is 1, ensure width and height are equal
            if aspectRatio == 1.0:
                effective_height = min(width, height[j])
                width = effective_height
            current_area = effective_height * width
            if current_area > max_area:
                max_area = current_area
                best_rect = (left[j], i - effective_height + 1, width, effective_height)
    return max_area, best_rect

def join_path(*path_components):
    """
    Join multiple path components into a single path using the appropriate separator for the operating system.

    Args:
        *path_components: Variable number of path components to be joined.

    Returns:
        str: Joined path string.

    """
    return os.path.join(*path_components)

def get_mask(data_path, dataset):
    """
    Build the full path to ``mask.png`` using ``data_path`` and ``dataset``,
    then read it with OpenCV.

    Args:
        data_path (str): Root directory of the dataset collection.
        dataset (str): Name of the specific dataset.

    Returns:
        numpy.ndarray | None: Loaded mask image, or ``None`` if the file does
        not exist or cannot be read.
    """
    # Build the path to mask.png
    mask_path = join_path(data_path, dataset, 'cam', 'mask', 'mask.png')
    print("Mask path:", mask_path)

    # Check the path and load the image
    if os.path.exists(mask_path):
        # Load the image with OpenCV
        im_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if im_mask is not None:
            print("Mask file loaded successfully.")
            return im_mask
        else:
            print("Failed to read the mask file.")
            return None
    else:
        print("Mask file does not exist.")
        return None

def calculate_bbox_from_mask(mask_image):
    # Convert mask_image to binary format (0 or 255)
    mask_binary = (mask_image > 0).astype(np.uint8)
    # Find contours of the mask
    contours, _ = cv.findContours(mask_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Find the minimum bounding rectangle of the largest contour
    bbox = cv.boundingRect(max(contours, key=cv.contourArea))
    return bbox

def place_image_in_rectangle_opencv(image, mask, rectangle):
    # Get the rectangle position and size
    x, y, width, height = rectangle
    # Resize the image to match the rectangle
    resized_image = cv.resize(image, (width, height))
    # Create a new image with the same size as the mask and initialize it as black
    # The mask should be a 2D array; here we create a matching 3-channel image
    new_image = np.zeros_like(mask)
    if len(mask.shape) == 2:  # Convert single-channel mask to three channels
        new_image = cv.cvtColor(new_image, cv.COLOR_GRAY2BGR)
    # Place the resized image into the target region
    new_image[y:y + height, x:x + width] = resized_image
    return new_image

def crop_image(image, bbox):
    """
    Crop an image using the provided bounding box.

    :param image: Input image (numpy array)
    :param bbox: Bounding box (x1, y1, w1, h1)
    :return: Cropped image (numpy array)
    """
    x1, y1, w1, h1 = bbox
    # Compute the crop area
    crop = image[y1:y1 + h1, x1:x1 + w1]
    return crop


# Assume these helper functions already exist
# from differential_color_function import * # from utils_all.pytorch_ssim import SSIM

# ==========================================
# Core metric computation function (strictly follows the original logic)
# ==========================================
def calculate_metrics_strict(pred, target):
    # Make sure inputs are tensors in the [0, 1] range
    if pred.max() > 1.0 or target.max() > 1.0:
        p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
        t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).float()

    p, t = p.to(device), t.to(device)

    with torch.no_grad():
        mae = l1_fun(p, t).item()
        mse = l2_fun(p, t).item()
        rmse = math.sqrt(mse)
        psnr = 10 * math.log10(1 / mse) if mse > 0 else 100
        ssim = ssim_fun(p, t).item()

        # Color difference
        xl = rgb2lab_diff(p, device)
        yl = rgb2lab_diff(t, device)
        deltaE = ciede2000_diff(xl, yl, device).mean().item()

        # LPIPS
        lp = loss_fn_lpips(p, t).mean().item()

        return mae, rmse, psnr, ssim, deltaE, lp


# ==========================================
# Main program logic
# ==========================================
def run_real_compensation_eval(data_path, data_list):
    all_summary = []

    for dataset in data_list:
        print(f"\n>>> Processing dataset: {dataset}")

        # Path definitions
        desired_GT_dir = os.path.join(data_path, dataset, 'cam', 'desire', "test")
        raw_cmp_root = os.path.join(data_path, dataset, 'prj', 'cmp')
        # Load the mask
        im_mask = get_mask(data_path, dataset)
        bbox = calculate_bbox_from_mask(im_mask)

        # Get the precomputed desired_mask used for bitwise_and
        # In the original logic, desired_mask is generated from full_best_rect
        _, full_best_rect = maximalRectangle((im_mask > 0).astype(np.uint8), aspectRatio=1)
        # Recreate the key mask
        h, w = im_mask.shape[:2]
        desired_mask = np.zeros((h, w), dtype=np.uint8)
        rx, ry, rw, rh = full_best_rect
        desired_mask[ry:ry + rh, rx:rx + rw] = 255

        # Traverse model folders
        cmp_folders = [f for f in os.listdir(raw_cmp_root) if os.path.isdir(os.path.join(raw_cmp_root, f))]

        for m_name in cmp_folders:
            raw_cmp_dir = os.path.join(data_path, dataset, 'cam', 'raw', m_name)
            # Temporary directory for FID computation
            fid_temp_dir = os.path.join(data_path, dataset, 'cam', 'desire', 'fid_temp', m_name)
            os.makedirs(fid_temp_dir, exist_ok=True)

            metrics_sum = {'mae': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0, 'deltaE': 0, 'lpips': 0}
            count = 0

            img_names = [f for f in os.listdir(raw_cmp_dir) if f.endswith('.png')]
            for name in tqdm(img_names, desc=f"Eval {m_name}"):
                p_path = os.path.join(raw_cmp_dir, name)
                t_path = os.path.join(desired_GT_dir, name)

                if not os.path.exists(t_path): continue

                # Read images in BGR format to match OpenCV-based metric logic
                img_raw = cv2.imread(p_path)
                img_gt = cv2.imread(t_path)

                # 1. Masking step (this matches the original logic)
                masked_raw = cv2.bitwise_and(img_raw, img_raw, mask=desired_mask)

                # Save images for FID (FID typically expects full-image distribution)
                cv2.imwrite(os.path.join(fid_temp_dir, name), masked_raw)

                # 2. Crop step (critical for alignment)
                crop_raw = crop_image(masked_raw, bbox)
                crop_gt = crop_image(img_gt, bbox)

                # 3. Compute metrics
                mae, rmse, psnr, ssim, dE, lp = calculate_metrics_strict(crop_raw, crop_gt)

                metrics_sum['mae'] += mae
                metrics_sum['rmse'] += rmse
                metrics_sum['psnr'] += psnr
                metrics_sum['ssim'] += ssim
                metrics_sum['deltaE'] += dE
                metrics_sum['lpips'] += lp
                count += 1

            # Compute FID
            print(f">>> Computing FID for {m_name}...")
            try:
                # FID compares masked camera captures against the masked reference images
                fid_val = fid_score.calculate_fid_given_paths(
                    [desired_GT_dir, fid_temp_dir],
                    batch_size=5, device=device, dims=2048, num_workers=0
                )
            except:
                fid_val = float('nan')

            if count > 0:
                all_summary.append({
                    'Model': m_name,
                    'Dataset': dataset,
                    'PSNR': metrics_sum['psnr'] / count,
                    'RMSE': metrics_sum['rmse'] / count,
                    'SSIM': metrics_sum['ssim'] / count,
                    'DeltaE': metrics_sum['deltaE'] / count,
                    'LPIPS': metrics_sum['lpips'] / count,
                    'FID': fid_val
                })

    # --- Final export ---
    save_final_excel(all_summary, data_path)


def save_final_excel(results, root):
    # 1. Create DataFrame
    df = pd.DataFrame(results)

    # 2. Compute the mean across all datasets
    # Do not call .round(4) here, otherwise Excel formatting will not take effect
    df_avg = df.groupby('Model').mean(numeric_only=True).reset_index()

    # 3. Replace DeltaE column name with the ΔE symbol
    df_avg = df_avg.rename(columns={'DeltaE': 'ΔE'})

    # 4. Ensure the column order is correct
    cols = ['Model', 'PSNR', 'RMSE', 'SSIM', 'ΔE', 'LPIPS', 'FID']
    # Only keep columns that actually exist
    existing_cols = [c for c in cols if c in df_avg.columns]
    df_avg = df_avg[existing_cols]

    path = os.path.join(root, "Actual_Final_Metrics.xlsx")

    # 5. Save with the xlsxwriter engine
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df_avg.to_excel(writer, sheet_name='Summary', index=False)

        workbook = writer.book
        worksheet = writer.sheets['Summary']

        # Force 4-decimal display format (0.0000)
        format_4_decimal = workbook.add_format({'num_format': '0.0000', 'align': 'center'})

        # Header format
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1,
            'align': 'center'
        })

        # Apply 4-decimal formatting to numeric columns
        num_cols = len(existing_cols)
        if num_cols > 1:
            worksheet.set_column(1, num_cols - 1, 15, format_4_decimal)

        # Rewrite header cells to apply the custom header style
        for col_num, value in enumerate(df_avg.columns.values):
            worksheet.write(0, col_num, value, header_format)

    print(f"✅ Done! File saved to {path}. Metrics are formatted to 4 decimals and ΔE is used as the column name.")

if __name__ == '__main__':
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize metric tools only once to save memory and time
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    ssim_fun = pytorch_ssim_fun().to(device)

    # 2. Read the root directory from an environment variable
    # data_root = os.getenv("DATASET_ROOT", r"E:\Desktop\SIComp_Benchmark\Real_datasets")
    data_root = os.getenv("DATASET_ROOT", r"E:\Desktop\SIComp_Benchmark\New_Real_datasets")
    # 3. Collect all valid dataset names under data_root
    data_list = get_linux_style_dataset_list(data_root)

    print(f"🚀 [Total Eval] Starting full evaluation on: {data_root}")
    print(f"📂 Datasets to process: {data_list}")

    # Run the evaluation once with the complete dataset list
    run_real_compensation_eval(data_root, data_list)