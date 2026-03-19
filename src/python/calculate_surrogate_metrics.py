from pytorch_fid import fid_score  # 重新引入 FID
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



# ==========================================
# 1. 核心计算函数
# ==========================================
def calculate_image_metrics(pred, target):
    with torch.no_grad():
        mae = l1_fun(pred, target).item()
        mse = l2_fun(pred, target).item()
        rmse = math.sqrt(mse)
        psnr = 10 * math.log10(1 / mse) if mse > 0 else 100
        ssim = ssim_fun(pred, target).item()

        # 色差计算
        xl_batch = rgb2lab_diff(pred, device)
        yl_batch = rgb2lab_diff(target, device)
        diff_map = ciede2000_diff(xl_batch, yl_batch, device).mean().item()

        # Lpips感官质量
        lpips_val = loss_fn_lpips(pred, target).mean().item()

        return mae, rmse, psnr, ssim, diff_map, lpips_val

# ==========================================
# 2. 文件夹对比逻辑 (PNG + FID 版)
# ==========================================
def compare_image_folders_offline(folder_pred, folder_gt, dataname, model_name):
    print(f">>> 正在计算 FID...")
    try:
        # FID 计算：对比预测文件夹和 GT 文件夹
        # batch_size 和 dims 可以根据显存调整，2048 是标准维度
        fid_val = fid_score.calculate_fid_given_paths(
            [folder_gt, folder_pred],
            batch_size=32,
            device=device,
            dims=2048,
            num_workers=0  # Windows下设为0更稳定
        )
    except Exception as e:
        print(f"!!! FID 计算失败: {e}")
        fid_val = float('nan')

    # 获取图像文件列表 (排除 .pt 和 tensors 文件夹)
    files_p = sorted([f for f in os.listdir(folder_pred) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    files_g = sorted([f for f in os.listdir(folder_gt) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 确保数量对齐
    if len(files_p) != len(files_g):
        min_len = min(len(files_p), len(files_g))
        files_p, files_g = files_p[:min_len], files_g[:min_len]
        print(f"⚠️ 警告: 数量不匹配，已截断为前 {min_len} 张进行对比")

    total_metrics = {'mae': 0, 'rmse': 0, 'psnr': 0, 'ssim': 0, 'deltaE': 0, 'lpips': 0}
    # 这里的 batch_size 决定了 PSNR/SSIM 是一次算几张图的平均
    calc_batch_size = 5
    batch_count = 0

    for i in tqdm(range(0, len(files_p), calc_batch_size), desc=f"计算指标 {dataname}"):
        b_p_names = files_p[i: i + calc_batch_size]
        b_g_names = files_g[i: i + calc_batch_size]

        batch_pred, batch_gt = [], []
        for p_name, g_name in zip(b_p_names, b_g_names):
            # 读取预测图 (PNG)
            img_p = cv2.cvtColor(cv2.imread(os.path.join(folder_pred, p_name)), cv2.COLOR_BGR2RGB)
            p_tensor = torch.from_numpy(img_p.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

            # 读取 GT 图
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

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    # --- 关键修改：从环境变量读取路径，读取不到则用默认值 ---
    dataset_root = Path(os.getenv("DATASET_ROOT", r'E:\Desktop\Surrogate_datasets'))
    excel_file_path = os.path.join(dataset_root, 'final_offline_results_with_fid.xlsx')
    data_list = get_linux_style_dataset_list(dataset_root)

    # 指定模型名称，None 则跑全部
    # target_model_name = "FF_PANet"
    target_model_name = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化度量工具
    l1_fun = nn.L1Loss().to(device)
    l2_fun = nn.MSELoss().to(device)
    ssim_fun = SSIM().to(device)
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

    all_results = []
    for dataname in data_list:
        # 注意：此处确保 GT 文件夹路径正确，通常 GT 在各自数据集文件夹下或统一位置
        # 根据你之前的逻辑，这里是统一的 gt 路径
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
                print(f">>> [跳过] 未找到路径: {m_path}")
                continue

            print(f"\n>>> 正在全面评估指标: {m_name} @ {dataname}")
            res = compare_image_folders_offline(m_path, gt_folder, dataname, m_name)
            all_results.append(res)

            # 清理显存防止 FID 累积溢出
            torch.cuda.empty_cache()

    # if all_results:
    #     # 1. 构建原始明细表 (Sheet1)
    #     df_details = pd.DataFrame(all_results)
    #     detail_cols = ['Model', 'Dataset', 'PSNR', 'RMSE', 'SSIM', 'DeltaE', 'LPIPS', 'FID', 'MAE']
    #     df_details = df_details[detail_cols].sort_values(by=['Model', 'Dataset'])
    #
    #     # 2. 构建平均统计表 (Sheet2)
    #     summary_cols = ['Model', 'PSNR', 'RMSE', 'SSIM', 'DeltaE', 'LPIPS', 'FID']
    #     df_summary = df_details.groupby('Model').mean(numeric_only=True).reset_index()
    #     df_summary = df_summary[summary_cols]
    #
    #     # 注意：这里虽然做了 round(4)，但 Excel 依然会隐藏末尾 0
    #     df_details = df_details.round(4)
    #     df_summary = df_summary.round(4)
    #
    #     # 3. 使用 ExcelWriter 写入
    #     with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    #         df_details.to_excel(writer, sheet_name='Detailed_Results', index=False)
    #         df_summary.to_excel(writer, sheet_name='Model_Average_Summary', index=False)
    #
    #         # --- 获取 workbook 对象进行格式定义 ---
    #         workbook = writer.book
    #
    #         # 定义表头格式
    #         header_format = workbook.add_format({
    #             'bold': True,
    #             'bg_color': '#D7E4BC',
    #             'border': 1,
    #             'align': 'center',
    #             'valign': 'vcenter'
    #         })
    #
    #         # 定义数值列格式：关键就在 'num_format': '0.0000'
    #         decimal_format = workbook.add_format({
    #             'num_format': '0.0000',
    #             'align': 'center',
    #             'valign': 'vcenter'
    #         })
    #
    #         # 定义文本列格式
    #         text_format = workbook.add_format({
    #             'align': 'center',
    #             'valign': 'vcenter'
    #         })
    #
    #         # 循环设置两个 Sheet 的显示效果
    #         for sheet_name in ['Detailed_Results', 'Model_Average_Summary']:
    #             worksheet = writer.sheets[sheet_name]
    #             # 获取当前 Sheet 对应的 DataFrame
    #             current_df = df_details if sheet_name == 'Detailed_Results' else df_summary
    #
    #             # 遍历所有列进行美化
    #             for i, col in enumerate(current_df.columns):
    #                 # 重新写入表头以覆盖默认格式
    #                 worksheet.write(0, i, col, header_format)
    #
    #                 if col in ['Model', 'Dataset']:
    #                     # 设置文本列：宽度25，应用居中格式
    #                     worksheet.set_column(i, i, 25, text_format)
    #                 else:
    #                     # 设置数值列：宽度12，应用强制4位小数格式
    #                     worksheet.set_column(i, i, 12, decimal_format)
    #
    #     print(f"\n✅ 评估完成！汇总报告已生成：{excel_file_path}")
    #     print(f"指标说明：数值已强制显示4位小数（如 30.5000）。")
    # ==========================================
    # 4. 最终保存逻辑 (仅保留平均汇总表)
    # ==========================================
    if all_results:
        # 1. 构建 DataFrame
        df_full = pd.DataFrame(all_results)

        # 2. 计算平均统计表
        summary_cols = ['Model', 'PSNR', 'RMSE', 'SSIM', 'DeltaE', 'LPIPS', 'FID']
        df_summary = df_full.groupby('Model').mean(numeric_only=True).reset_index()

        # 3. 列名处理：将 DeltaE 替换为罗马符号 ΔE
        df_summary = df_summary.rename(columns={'DeltaE': 'ΔE'})

        # 确保列顺序正确 (使用新列名)
        final_cols = ['Model', 'PSNR', 'RMSE', 'SSIM', 'ΔE', 'LPIPS', 'FID']
        existing_cols = [c for c in final_cols if c in df_summary.columns]
        df_summary = df_summary[existing_cols]

        # 4. 使用 ExcelWriter 写入单一 Sheet
        with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
            # 只写入一个名为 'Model_Average_Summary' 的表
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

            # --- 格式化设置 ---
            workbook = writer.book
            worksheet = writer.sheets['Summary']

            # 定义表头格式 (加粗、背景色、居中)
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BC',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            # 定义数值列格式：强制 4 位小数
            decimal_format = workbook.add_format({
                'num_format': '0.0000',
                'align': 'center',
                'valign': 'vcenter'
            })

            # 遍历列应用格式
            for i, col in enumerate(df_summary.columns):
                # 重新写入表头以应用样式
                worksheet.write(0, i, col, header_format)

                if col == 'Model':
                    # 模型名称列加宽
                    worksheet.set_column(i, i, 25, workbook.add_format({'align': 'center'}))
                else:
                    # 指标数值列应用 4 位小数格式
                    worksheet.set_column(i, i, 12, decimal_format)

        print(f"\n✅ 评估完成！汇总报告已生成：{excel_file_path}")
        print(f"统计说明：仅包含模型平均值，DeltaE 已重命名为 ΔE，数值强制补齐 4 位小数。")
