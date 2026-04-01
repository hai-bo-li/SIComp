from pytorch_fid import fid_score
from utils_all.differential_color_function import *
from utils_all.pytorch_ssim import *
from utils_all.utils import get_linux_style_dataset_list
import math
import os
import torch.nn as nn
import pandas as pd
import cv2
import lpips
from pathlib import Path
from tqdm import tqdm


def calculate_image_metrics(pred, target):
    with torch.no_grad():
        mae = l1_fun(pred, target).item()
        mse = l2_fun(pred, target).item()
        rmse = math.sqrt(mse)
        psnr = 10 * math.log10(1 / mse) if mse > 0 else 100
        ssim = ssim_fun(pred, target).item()

        # Delta E calculation
        xl_batch = rgb2lab_diff(pred, device)
        yl_batch = rgb2lab_diff(target, device)
        diff_map = ciede2000_diff(xl_batch, yl_batch, device).mean().item()

        # LPIPS perceptual quality
        lpips_val = loss_fn_lpips(pred, target).mean().item()

        return mae, rmse, psnr, ssim, diff_map, lpips_val


def compare_image_folders_offline(folder_pred, folder_gt, dataname, model_name):
    print(f">>> Calculating FID...")
    try:
        # FID calculation: compare the prediction folder and the GT folder
        # batch_size and dims can be adjusted according to GPU memory; 2048 is the standard dimension
        fid_val = fid_score.calculate_fid_given_paths(
            [folder_gt, folder_pred],
            batch_size=32,
            device=device,
            dims=2048,
            num_workers=0  # Set to 0 on Windows for better stability
        )
    except Exception as e:
        print(f"!!! FID calculation failed: {e}")
        fid_val = float('nan')

    # Get image file lists (excluding .pt and tensors folders)
    files_p = sorted([f for f in os.listdir(folder_pred) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    files_g = sorted([f for f in os.listdir(folder_gt) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Ensure the counts are aligned
    if len(files_p) != len(files_g):
        min_len = min(len(files_p), len(files_g))
        files_p, files_g = files_p[:min_len], files_g[:min_len]
        print(f"⚠️ Warning: file counts do not match; truncated to the first {min_len} images for comparison")

    total_metrics = {'mae': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0, 'deltaE': 0, 'lpips': 0}
    # This batch size determines how many images are averaged together per PSNR/SSIM computation step
    calc_batch_size = 5
    batch_count = 0

    for i in tqdm(range(0, len(files_p), calc_batch_size), desc=f"Computing metrics for {dataname}"):
        b_p_names = files_p[i: i + calc_batch_size]
        b_g_names = files_g[i: i + calc_batch_size]

        batch_pred, batch_gt = [], []
        for p_name, g_name in zip(b_p_names, b_g_names):
            # Read the prediction image (PNG)
            img_p = cv2.cvtColor(cv2.imread(os.path.join(folder_pred, p_name)), cv2.COLOR_BGR2RGB)
            p_tensor = torch.from_numpy(img_p.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

            # Read the GT image
            img_g = cv2.cvtColor(cv2.imread(os.path.join(folder_gt, g_name)), cv2.COLOR_BGR2RGB)
            g_tensor = torch.from_numpy(img_g.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

            batch_pred.append(p_tensor)
            batch_gt.append(g_tensor)

        p_batch = torch.cat(batch_pred, dim=0).to(device)
        g_batch = torch.cat(batch_gt, dim=0).to(device)

        mae, rmse, psnr, ssim, dE, lp = calculate_image_metrics(p_batch, g_batch)

        total_metrics['mae'] += mae
        total_metrics['rmse'] += rmse
        total_metrics['psnr'] += psnr
        total_metrics['ssim'] += ssim
        total_metrics['deltaE'] += dE
        total_metrics['lpips'] += lp
        batch_count += 1

    return {
        'Model': model_name, 'Dataset': dataname,
        'PSNR': total_metrics['psnr'] / batch_count,
        'SSIM': total_metrics['ssim'] / batch_count,
        'LPIPS': total_metrics['lpips'] / batch_count,
        'FID': fid_val,
        'DeltaE': total_metrics['deltaE'] / batch_count,
        'MAE': total_metrics['mae'] / batch_count,
        'RMSE': total_metrics['rmse'] / batch_count
    }


if __name__ == '__main__':
    dataset_root = Path(os.getenv("DATASET_ROOT", r'E:\Desktop\Surrogate_datasets'))
    excel_file_path = os.path.join(dataset_root, 'final_offline_results_with_fid.xlsx')
    data_list = get_linux_style_dataset_list(dataset_root)

    # Specify the model name; use None to run all models
    # target_model_name = "FF_PANet"
    target_model_name = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize metric tools
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    ssim_fun = SSIM().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

    all_results = []
    for dataname in data_list:
        # Make sure the GT folder path is correct here. Usually the GT data is stored
        # either under each dataset folder or in a shared location.
        # According to the current logic, the GT path is shared here.
        gt_folder = os.path.join(dataset_root, 'test')
        test_models_root = os.path.join(dataset_root, dataname, 'prj', 'test')

        if not os.path.isdir(test_models_root):
            continue

        if target_model_name is not None:
            model_folders = [target_model_name]
        else:
            model_folders = [d for d in os.listdir(test_models_root)
                             if os.path.isdir(os.path.join(test_models_root, d)) and d != 'tensors']

        for m_name in model_folders:
            m_path = os.path.join(test_models_root, m_name)
            if not os.path.exists(m_path):
                print(f">>> [Skip] Path not found: {m_path}")
                continue

            print(f"\n>>> Running full metric evaluation: {m_name} @ {dataname}")
            res = compare_image_folders_offline(m_path, gt_folder, dataname, m_name)
            all_results.append(res)

            # Clear GPU memory to avoid FID-related accumulation
            torch.cuda.empty_cache()

    if all_results:
        # 1. Build the DataFrame
        df_full = pd.DataFrame(all_results)

        # 2. Compute the summary table
        summary_cols = ['Model', 'PSNR', 'RMSE', 'SSIM', 'DeltaE', 'LPIPS', 'FID']
        df_summary = df_full.groupby('Model').mean(numeric_only=True).reset_index()

        # 3. Rename DeltaE to the symbol ΔE
        df_summary = df_summary.rename(columns={'DeltaE': 'ΔE'})

        # Ensure the column order is correct (using the new column name)
        final_cols = ['Model', 'PSNR', 'RMSE', 'SSIM', 'ΔE', 'LPIPS', 'FID']
        existing_cols = [c for c in final_cols if c in df_summary.columns]
        df_summary = df_summary[existing_cols]

        # 4. Write a single sheet with ExcelWriter
        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            # Write only one sheet named 'Summary'
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

            # --- Formatting setup ---
            workbook = writer.book
            worksheet = writer.sheets['Summary']

            # Define the header format (bold, background color, centered)
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BC',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            # Define the numeric column format: force 4 decimal places
            decimal_format = workbook.add_format({
                'num_format': '0.0000',
                'align': 'center',
                'valign': 'vcenter'
            })

            # Apply formatting column by column
            for i, col in enumerate(df_summary.columns):
                # Rewrite the header to apply the style
                worksheet.write(0, i, col, header_format)

                if col == 'Model':
                    # Widen the model name column
                    worksheet.set_column(i, i, 25, workbook.add_format({'align': 'center'}))
                else:
                    # Apply the 4-decimal format to metric columns
                    worksheet.set_column(i, i, 12, decimal_format)

        print(f"\n✅ Evaluation completed! Summary report generated at: {excel_file_path}")
        print(f"Summary note: only model averages are included, DeltaE has been renamed to ΔE, and values are padded to 4 decimal places.")
